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

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.2, device='cuda'):
    model.eval()
    
    stop_token_id = tokenizer.token_to_id("<|end|>")
    
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoding.ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > model.config.max_seq_len:
                input_ids = input_ids[:, -model.config.max_seq_len:]
            
            logits, _ = model(input_ids)
            logits = logits[:, -1, :].float() 
        
            if repetition_penalty != 1.0:
                for i in range(input_ids.size(0)):
                    for token_id in set(input_ids[i].tolist()):
                        logits[i, token_id] /= repetition_penalty

            logits = logits / (temperature if temperature > 0 else 1.0)
            
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
                logits[:, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == stop_token_id:
                break
            
            print(tokenizer.decode([next_token.item()]), end="", flush=True)

            input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=1000)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--rep_penalty', type=float, default=1.2)
    parser.add_argument('--tokenizer', type=str, default='tokenizer.json')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)

    config = GPTConfig(
        vocab_size=32064, 
        embed_dim=768,    
        n_layers=12,
        n_heads=12,
        num_experts=4,
        top_k=2
    )
    
    model = GPT(config)
    load_model(model, args.checkpoint)
    model.to(args.device)

    system_prompt = "<|system|> Ты Vexion-LM, опытный инженер и ИИ-ассистент. Отвечай технически грамотно. <|end|>\n"
    
    print("Vexion-LM готова. Введи запрос:")
    
    while True:
        user_input = input("\n Юзер: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        full_prompt = f"{system_prompt}<|user|> {user_input} <|end|>\n<|assistant|> "
        
        print("Vexion-LM: ", end="", flush=True)
        generate(
            model, tokenizer, full_prompt, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            device=args.device
        )
        print()
