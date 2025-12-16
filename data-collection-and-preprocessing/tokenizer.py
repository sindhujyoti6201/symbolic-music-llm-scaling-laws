#!/usr/bin/env python3
"""
Music Tokenizer
Tokenizer for ABC notation with music-aware tokens.
"""

import pickle
from pathlib import Path
from collections import Counter
from typing import List
from tqdm import tqdm


class MusicTokenizer:
    """Tokenizer for ABC notation."""
    
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
            '<SEP>': 4,
        }
    
    def build_vocab(self, abc_strings: List[str], min_freq: int = 2):
        """Build vocabulary from ABC strings."""
        print("Building vocabulary...")
        token_counter = Counter()
        
        for abc_str in tqdm(abc_strings, desc="Tokenizing for vocab"):
            tokens = self._tokenize_abc(abc_str)
            token_counter.update(tokens)
        
        vocab = dict(self.special_tokens)
        current_id = len(self.special_tokens)
        
        for token, count in token_counter.items():
            if count >= min_freq:
                vocab[token] = current_id
                current_id += 1
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"  Special tokens: {len(self.special_tokens)}")
        print(f"  Regular tokens: {self.vocab_size - len(self.special_tokens)}")
    
    def _tokenize_abc(self, abc_str: str) -> List[str]:
        """Tokenize ABC notation string into music-aware tokens."""
        tokens = []
        i = 0
        
        while i < len(abc_str):
            char = abc_str[i]
            
            if char.isspace():
                i += 1
                continue
            
            if char == '|':
                tokens.append('|')
                i += 1
                continue
            
            if char.upper() in 'ABCDEFG':
                note_token = char.upper()
                i += 1
                
                if i < len(abc_str) and abc_str[i] in '^_':
                    note_token += abc_str[i]
                    i += 1
                
                while i < len(abc_str) and abc_str[i] in ",'":
                    note_token += abc_str[i]
                    i += 1
                
                tokens.append(note_token)
                continue
            
            if char.isdigit():
                duration = char
                i += 1
                while i < len(abc_str) and abc_str[i].isdigit():
                    duration += abc_str[i]
                    i += 1
                tokens.append(f"DUR:{duration}")
                continue
            
            if char == 'z':
                tokens.append('z')
                i += 1
                continue
            
            tokens.append(char)
            i += 1
        
        return tokens
    
    def encode(self, abc_str: str) -> List[int]:
        """Encode ABC string to token IDs."""
        tokens = self._tokenize_abc(abc_str)
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id['<UNK>'])
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to ABC string."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<UNK>')
        return ' '.join(tokens)
    
    def save(self, path: Path):
        """Save tokenizer to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }, f)
    
    @classmethod
    def load(cls, path: Path):
        """Load tokenizer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tokenizer = cls()
        tokenizer.vocab = data['vocab']
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = data['id_to_token']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.special_tokens = data['special_tokens']
        return tokenizer

