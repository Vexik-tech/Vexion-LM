# Copyright 2026 Dmitry
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import glob
import pickle
import bitsandbytes as bnb
from tqdm import tqdm
from safetensors.torch import save_model, load_model
from model import GPT, GPTConfig
from tokenizer import train_tokenizer, load_tokenizer
import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel

class FastDataloader:
    def __init__(self, bin_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        print(f"✅ Базовый датасет загружен. Всего токенов: {len(self.data):,}")

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.data) - self.max_seq_len - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.max_seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.max_seq_len].astype(np.int64)) for i in ix])
        return x, y

class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, max_seq_len):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        self.user_tok = tokenizer.encode("<|user|>").ids[0]
        self.assist_tok = tokenizer.encode("<|assistant|>").ids[0]
        self.end_tok = tokenizer.encode("<|end|>").ids[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        try:
            user_text, bot_text = line.split(" | ")
        except ValueError:
            user_text, bot_text = "Ошибка", "Используйте разделитель |"
            
        prompt_ids = self.tokenizer.encode(user_text.strip()).ids
        response_ids = self.tokenizer.encode(bot_text.strip()).ids
        
        x_ids = [self.user_tok] + prompt_ids + [self.end_tok, self.assist_tok] + response_ids + [self.end_tok]
        
        ignore_len = 1 + len(prompt_ids) + 1 + 1 
        y_ids = [-100] * ignore_len + response_ids + [self.end_tok]
        
        if len(x_ids) > self.max_seq_len:
            x_ids = x_ids[:self.max_seq_len]
            y_ids = y_ids[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(x_ids)
            x_ids = x_ids + [0] * pad_len
            y_ids = y_ids + [-100] * pad_len 
            
        return torch.tensor(x_ids, dtype=torch.long), torch.tensor(y_ids, dtype=torch.long)

@torch.no_grad()
def validate(model, val_loader, batch_size, eval_iters=50):
    print("DEBUG: starting fast validation...")
    model.eval()
    
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        x, y = val_loader.get_batch(batch_size)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            logits, loss = model(x, y)
            
        losses[k] = loss.item()
        
    model.train()
    avg_loss = losses.mean().item()
    print(f"DEBUG: validation finished, avg_loss = {avg_loss:.4f}")
    return avg_loss
                
def line_generator(file_path, max_lines=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            yield line

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)

def train(args):
    print(f"DEBUG: args.val_path = {args.val_path}")
    print(f"DEBUG: file exists? {os.path.exists(args.val_path) if args.val_path else False}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if os.path.exists(args.tokenizer_path):
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = load_tokenizer(args.tokenizer_path)
    else:
        print("Tokenizer not found. Training new tokenizer...")
        train_tokenizer(line_generator(args.data_path), 
                        vocab_size=args.vocab_size, save_path=args.tokenizer_path)
        tokenizer = load_tokenizer(args.tokenizer_path)

    if not args.use_lora:
        print(f" Режим БАЗЫ. Загрузка бинарника: {args.data_path}")
        train_loader = FastDataloader(args.data_path, args.max_seq_len)
        def get_train_batch(): 
            return train_loader.get_batch(args.batch_size)
            
        get_val_batch = None
        if args.val_path and os.path.exists(args.val_path):
            val_loader = FastDataloader(args.val_path, args.max_seq_len)
            def get_val_batch(): 
                return val_loader.get_batch(args.batch_size)
    else:
        print(f" Режим DoRA. Загрузка txt диалогов: {args.data_path}")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            chat_data = [line.strip() for line in f if line.strip()]
            
        chat_dataset = ChatDataset(chat_data, tokenizer, args.max_seq_len)
        chat_loader = DataLoader(chat_dataset, batch_size=args.batch_size, shuffle=True)
        chat_iter = iter(chat_loader)
        
        def get_train_batch():
            nonlocal chat_iter
            try:
                x, y = next(chat_iter)
            except StopIteration:
                chat_iter = iter(chat_loader)
                x, y = next(chat_iter)
            return x, y
            
        get_val_batch = None
        if args.val_path and os.path.exists(args.val_path):
            with open(args.val_path, 'r', encoding='utf-8') as f:
                val_chat_data = [line.strip() for line in f if line.strip()]
            val_chat_dataset = ChatDataset(val_chat_data, tokenizer, args.max_seq_len)
            val_chat_loader = DataLoader(val_chat_dataset, batch_size=args.batch_size, shuffle=True)
            val_chat_iter = iter(val_chat_loader)
            
            def get_val_batch():
                nonlocal val_chat_iter
                try:
                    x, y = next(val_chat_iter)
                except StopIteration:
                    val_chat_iter = iter(val_chat_loader)
                    x, y = next(val_chat_iter)
                return x, y

    model_config = GPTConfig(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        use_lora=args.use_lora
    )
    global_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = GPT(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("="*40)
    print(f"📊 Архитектура модели:")
    print(f"   Всего параметров:     {total_params:,}")
    print(f"   Обучаемых параметров: {trainable_params:,}")
    print("="*40)

    # Если указан чекпоинт, загружаем веса модели
    #if args.resume and os.path.exists(args.resume):
    #    print(f"Loading model from {args.resume}")
    #    load_model(model, args.resume)

    if args.use_lora:
        print("Включен режим LoRA: заморозка базовых весов..")
        for param in model.parameters():
            param.requires_grad = False
            
        lora_params = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if 'lora_A' in name or 'lora_B' in name or 'lora_m' in name:
                param.requires_grad = True
                lora_params += param.numel()
                
        print(f"Всего параметров: {total_params:,}")
        print(f"Обучаемые параметры LoRA: {lora_params:,} ({(lora_params/total_params)*100:.2f}%)")
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        print("🚀 Режим базового обучения: тренируем все параметры с нуля.")
        trainable_params = model.parameters()

    optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_steps, min_lr_ratio=0.1)
    
    if args.resume and getattr(args, 'use_lora', False):
        try:
            resume_step = int(os.path.basename(args.resume).split('_')[-1].split('.')[0])
            if resume_step < args.total_steps:
                scheduler.last_epoch = resume_step
                scheduler._step_count = resume_step + 1
        except:
            pass
    
    use_scaler = not torch.cuda.is_bf16_supported()
    if use_scaler:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading model weights from {args.resume}")
        
        from safetensors.torch import load_file
        sd = load_file(args.resume)
        
        if getattr(args, 'use_lora', False):
            new_sd = {}
            for k, v in sd.items():
                k = k.replace('c_attn.weight', 'c_attn.linear.weight')
                k = k.replace('c_attn.bias', 'c_attn.linear.bias')
                k = k.replace('c_proj.weight', 'c_proj.linear.weight')
                k = k.replace('c_proj.bias', 'c_proj.linear.bias')
                k = k.replace('c_fc.weight', 'c_fc.linear.weight')
                k = k.replace('c_fc.bias', 'c_fc.linear.bias')
                new_sd[k] = v
            model.load_state_dict(new_sd, strict=False)
        else:
            model.load_state_dict(sd, strict=False)
        
        print("✅ Weights successfully loaded!")

        import gc
        del sd
        if 'new_sd' in locals():
            del new_sd
        gc.collect()
        torch.cuda.empty_cache()

        opt_path = args.resume.replace('.safetensors', '.pt')
        if os.path.exists(opt_path) and not getattr(args, 'use_lora', False):
            print(f"Loading optimizer state from {opt_path}")
            try:
                with open(opt_path, 'rb') as f:
                    opt_state = pickle.load(f)
                optimizer.load_state_dict(opt_state['optimizer'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                scheduler.load_state_dict(opt_state['scheduler'])
                scheduler.base_lrs = [args.lr for _ in optimizer.param_groups]
                start_step = opt_state['step']
                
                print(f"✅ Optimizer loaded! Starting from step {start_step}")

                del opt_state
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"⚠️ Optimizer file corrupted ({e}). Starting optimizer from scratch.")
                try:
                    start_step = int(os.path.basename(args.resume).split('_')[-1].split('.')[0])
                except:
                    start_step = 0
        else:
            if getattr(args, 'use_lora', False):
                print("⚠️ LoRA mode: Optimizer state skipped, starting from scratch for adapters.")
                try:
                    start_step = int(os.path.basename(args.resume).split('_')[-1].split('.')[0])
                except:
                    start_step = 0
            else:
                print("⚠️ Optimizer state not found, starting optimizer from scratch.")
                try:
                    start_step = int(os.path.basename(args.resume).split('_')[-1].split('.')[0])
                except:
                    start_step = 0
            print(f"✅ Starting from step {start_step}")

    os.makedirs(args.save_dir, exist_ok=True)
    step = start_step
    best_loss = float('inf')
    best_val_loss = float('inf')

    model.train()
    progress_bar = tqdm(total=args.total_steps, initial=step, desc="Training")
    #data_iter = iter(dataloader)

    optimizer.zero_grad()
    
    micro_step = 0 
    accum_loss = 0.0 

    train_loader = FastDataloader(args.data_path, args.max_seq_len)
    val_loader = FastDataloader(args.val_path, args.max_seq_len)
    print("Начинаем обучение...")
    try:
        while step < args.total_steps:
            x, y = train_loader.get_batch(args.batch_size)
            
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                with torch.amp.autocast('cuda', dtype=global_dtype):
                    logits, loss = model(x, y)
                    loss = loss / args.accumulate_steps
            
            accum_loss += loss.item() 
            
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_step += 1 

            if micro_step % args.accumulate_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                step += 1
                progress_bar.update(1)

                if step % 100 == 0:  
                    total_mutated = 0
                    for layer in model.transformer.h: 
                        if hasattr(layer, 'moe'): 
                            mutated = layer.moe.mutate_dead_experts(optimizer)
                            total_mutated += mutated
                                    
                    if total_mutated > 0:
                        progress_bar.write(f" [Шаг {step}] Заменено мертвых экспертов -> {total_mutated}")
                
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                
                progress_bar.set_postfix(
                    loss=accum_loss, 
                    lr=optimizer.param_groups[0]['lr'],
                    vram=f"{allocated:.1f}G/{reserved:.1f}G"
                )
                accum_loss = 0.0

                if step % 200 == 0:
                    torch.cuda.empty_cache()

                if step % args.save_every == 0:
                    ckpt_path = os.path.join(args.save_dir, f"gpt_step_{step}.safetensors")
                    save_model(model, ckpt_path)
                    
                    opt_path = ckpt_path.replace('.safetensors', '.pt')
                    with open(opt_path, 'wb') as f:
                        pickle.dump({
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'step': step
                        }, f)
                    print(f"\nSaved checkpoint to {ckpt_path} and optimizer state")

                    if args.val_path and os.path.exists(args.val_path):
                        print(f"Running validation at step {step}...")
                        val_loss = validate(
                            model,
                            val_loader,
                            args.batch_size,
                            eval_iters=50
                        )
                        print(f"Step {step}: val loss = {val_loss:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_path = os.path.join(args.save_dir, "gpt_best.safetensors")
                            save_model(model, best_path)
                            print(f"New best model saved with val loss {val_loss:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано вручную (Ctrl+C)! Переходим к сохранению...")

    print(" Сохраняем финальную модель...")
    final_path = os.path.join(args.save_dir, "gpt_final.safetensors")
    state_dict = model.state_dict()

    if 'lm_head.weight' in state_dict and 'transformer.wte.weight' in state_dict:
        if (state_dict['lm_head.weight'].data_ptr() == state_dict['transformer.wte.weight'].data_ptr()):
            state_dict['lm_head.weight'] = state_dict['lm_head.weight'].clone()

    save_model(model, final_path)
    print(f"Model saved to {final_path}")

    opt_path = final_path.replace('.safetensors', '.pt')
    with open(opt_path, 'wb') as f:
        pickle.dump({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'step': step
        }, f)
    print(f"Optimizer state saved to {opt_path}")
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer.json')
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)  
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--total_steps', type=int, default=100000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--accumulate_steps', type=int, default=8, help="Шагов накопления для виртуального батча")
    parser.add_argument('--use_lora', action='store_true', help="Включить адаптацию через LoRA")
    args = parser.parse_args()
    train(args)
