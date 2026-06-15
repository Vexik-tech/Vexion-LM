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
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def train_tokenizer(texts, vocab_size=40960, save_path="tokenizer.json"): # Поставил правильный дефолт
    """
    Обучает BPE токенизатор на списке текстов и сохраняет в файл.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Punctuation(),
        pre_tokenizers.ByteLevel(add_prefix_space=True)
    ])
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", 
            "<|system|>", "<|user|>", "<|model|>", "<|endoftext|>", 
            "<|assistant|>", "<|end|>", "<|search|>", "<|search_end|>", 
            "<|result|>", "<|result_end|>", "<|thinking|>", "<|thinking_end|>", 
            "<|tool_call|>", "<|tool_result|>", "[code]", "[/code]"
        ],
        min_frequency=2,
        limit_alphabet=1000, 
        show_progress=True
    )
    
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    tokenizer.save(save_path)
    print(f"✅ Tokenizer successfully saved to {save_path} with vocab_size {vocab_size}")
    return tokenizer

def load_tokenizer(path="tokenizer.json"):
    return Tokenizer.from_file(path)

def tokenize_texts(tokenizer, texts, max_length, pad_token_id=0):
    encoding = tokenizer.encode_batch(texts)
    input_ids = [enc.ids[:max_length] for enc in encoding]
    return input_ids
