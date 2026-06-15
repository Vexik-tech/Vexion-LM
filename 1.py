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
from safetensors.torch import load_model, save_model
from model import GPT, GPTConfig

# Размеры словаря (строго кратно 64)
OLD_VOCAB = 34000
NEW_VOCAB = 34064 
checkpoint_path = "checkpoints/gpt_step_10000.safetensors"

old_config = GPTConfig(
    vocab_size=OLD_VOCAB,
    embed_dim=768,       # ЗАМЕНИ НА СВОЕ ЗНАЧЕНИЕ (Например, 384 или 768)
    n_layers=12,         # ЗАМЕНИ НА СВОЕ ЗНАЧЕНИЕ
    n_heads=12           # ЗАМЕНИ НА СВОЕ ЗНАЧЕНИЕ
)
old_model = GPT(old_config)
load_model(old_model, checkpoint_path)
print(f"✅ Старая модель загружена. Текущий словарь: {OLD_VOCAB}")

new_config = GPTConfig(
    vocab_size=NEW_VOCAB,
    embed_dim=old_config.embed_dim,
    n_layers=old_config.n_layers,
    n_heads=old_config.n_heads
)
new_model = GPT(new_config)

new_state = new_model.state_dict()
old_state = old_model.state_dict()

for name, param in old_state.items():
    if name in new_state and param.shape == new_state[name].shape:
        new_state[name].copy_(param)

print("📋 Внутренние веса архитектуры успешно скопированы.")

old_wte = old_state['transformer.wte.weight']
new_wte = new_state['transformer.wte.weight']
new_wte[:OLD_VOCAB] = old_wte 
print(f"🔧 Входные эмбеддинги (wte) расширены: {old_wte.shape[0]} -> {new_wte.shape[0]}")

if 'lm_head.weight' in old_state:
    old_lm = old_state['lm_head.weight']
    new_lm = new_state['lm_head.weight']
    new_lm[:OLD_VOCAB] = old_lm
    print(f"🔧 Выходной слой (lm_head) расширен: {old_lm.shape[0]} -> {new_lm.shape[0]}")

new_model.load_state_dict(new_state)
output_path = "checkpoints/gpt_extended_step_10000.safetensors"
save_model(new_model, output_path)

print(f"💾 Модель готова и сохранена в: {output_path}")
