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


import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPT, GPTConfig
from safetensors.torch import load_model  
import argparse
import os

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.2, device='cuda'):
    model.eval()
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoding.ids, dtype=torch.long, device=device).unsqueeze(0)

    prompt_length = input_ids.size(1)
    past_key_values = None  # Инициализируем наш новый KV-кэш

    with torch.no_grad():
        # Умный выбор устройства для autocast, чтобы не падало на CPU
        autocast_device = 'cuda' if 'cuda' in device else 'cpu'
        with torch.amp.autocast(autocast_device, enabled=(autocast_device == 'cuda')):
            for _ in range(max_new_tokens):
                
                # Если кэш уже есть, скармливаем модели только один ПОСЛЕДНИЙ токен
                if past_key_values is not None:
                    current_input = input_ids[:, -1:]
                else:
                    current_input = input_ids

                # Защита от переполнения контекста
                if current_input.size(1) > model.config.max_seq_len:
                    current_input = current_input[:, -model.config.max_seq_len:]
                
                # Вызываем обновленную модель с use_cache=True
                out = model(current_input, use_cache=True, past_key_values=past_key_values)
                logits = out[0]
                past_key_values = out[2] # Сохраняем обновленный кэш
                
                # Берем только последнее предсказание
                logits = logits[:, -1, :].float() 
                
                # Штраф за повторения
                if repetition_penalty != 1.0:
                    seen_tokens = torch.unique(input_ids)
                    score = logits[0, seen_tokens]
                    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits[0, seen_tokens] = score

                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Умный стоп-кран
                generated_tail_ids = input_ids[0, prompt_length:].tolist()
                tail_text = tokenizer.decode(generated_tail_ids)
                
                if "<|endoftext|>" in tail_text:
                    break

    final_tail_ids = input_ids[0, prompt_length:].tolist()
    final_text = tokenizer.decode(final_tail_ids)
    
    if "<|endoftext|>" in final_text:
        final_text = final_text.split("<|endoftext|>")[0]

    return prompt + final_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--rep_penalty', type=float, default=1.2)
    parser.add_argument('--tokenizer', type=str, default='tokenizer.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)

    # ОБНОВЛЕННЫЙ КОНФИГ ДЛЯ VEXION-LM MINI
    config = GPTConfig(
        vocab_size=40960,   # Новый размер словаря
        embed_dim=768,      
        n_layers=12,        
        n_heads=12,         
        max_seq_len=1024,   
        dropout=0.0,
        num_experts=4,      # Параметры новой архитектуры MoE
        top_k=2
    )
    
    model = GPT(config)
    load_model(model, args.checkpoint, strict=False)
    model.to(args.device)

    print("Генерация текста...")
    output = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
        device=args.device
    )
    print("\n" + "="*50)
    print("Результат:")
    print("="*50)
    print(output)
