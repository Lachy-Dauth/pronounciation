import random
import pandas as pd
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import math

STANDARD_IPA = [
    # Vowels
    'ɑ', 'æ', 'ʌ', 'ɔ', 'aʊ', 'aɪ', 'ɛ', 'ɝ', 'eɪ', 'ɪ', 'i', 'oʊ', 'ɔɪ', 'ʊ', 'u',
    # Consonants
    'b', 'tʃ', 'd', 'ð', 'f', 'ɡ', 'h', 'dʒ', 'k', 'l', 'm', 'n', 'ŋ', 'p', 'ɹ', 's', 'ʃ', 't', 'θ', 'v', 'w', 'j', 'z', 'ʒ',
    # Other
    'ʔ', 'ɾ', 'x',
    # Stress markers
    'ˈ', 'ˌ',
]

class PhoneDataset(Dataset):

    def __init__(self, data_path: str, max_words: int = 200000, max_len: int = 48):
        self.max_len = max_len
        self.data = []
        self.char_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.phone_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx_to_char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.idx_to_phone = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        
        for phone in STANDARD_IPA:
            if phone not in self.phone_to_idx:
                idx = len(self.phone_to_idx)
                self.phone_to_idx[phone] = idx
                self.idx_to_phone[idx] = phone

        self.direction = 'g2p'
        
        # Parse data file
        raw_data = self._parse_data(data_path)
        
        # Limit number of words, shuffling first to get a diverse sample
        if len(raw_data) > max_words:
            random.shuffle(raw_data)
            raw_data = raw_data[:max_words]
        
        # Build vocabularies from the loaded data
        self._build_vocabularies(raw_data)
        
        # Convert text and phonemes to numerical indices
        self.data = self._convert_to_indices(raw_data)
        
        print(f"Loaded {len(self.data)} word-pronunciation pairs.")
        print(f"Character vocab size: {len(self.char_to_idx)}")
        print(f"Phone vocab size: {len(self.phone_to_idx)}")
    
    def _parse_data(self, data_path: str) -> List[Tuple[str, List[str]]]:
        """Parse IPA pronunciation dictionary (format: word\t/ipa/)"""
        data = []
        # Sort known phonemes by length (desc) for greedy tokenization
        known_phonemes = sorted([p for p in self.phone_to_idx if len(p) > 1], key=len, reverse=True)

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '\t' not in line:
                    continue
                
                word, pronunciations = line.split('\t', 1)
                word = word.strip().upper()

                if not any(c.isalpha() for c in word):
                    continue

                # Take the first pronunciation and clean it
                pron = pronunciations.split(',')[0].strip().replace('/', '')

                # Tokenize IPA string into a list of phonemes
                phones = []
                i = 0
                while i < len(pron):
                    # Greedily match longest known multi-character phoneme first
                    match = next((p for p in known_phonemes if pron.startswith(p, i)), None)
                    
                    if match:
                        phones.append(match)
                        i += len(match)
                    else:
                        # Otherwise, take a single character
                        phones.append(pron[i])
                        i += 1
                
                data.append((word, phones))
        return data

    def _build_vocabularies(self, data: List[Tuple[str, List[str]]]):
        """Build character and phone vocabularies from the data"""
        chars = set()
        phones = set()
        
        for word, phone_seq in data:
            chars.update(word.lower())
            phones.update(phone_seq)
        
        # Add characters to vocab
        for char in sorted(list(chars)):
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        
        # Add phones to vocab
        for phone in sorted(list(phones)):
            if phone not in self.phone_to_idx:
                idx = len(self.phone_to_idx)
                self.phone_to_idx[phone] = idx
                self.idx_to_phone[idx] = phone
    
    def _convert_to_indices(self, data: List[Tuple[str, List[str]]]) -> List[Tuple[List[int], List[int]]]:
        """Convert words and phones to padded index sequences"""
        converted = []
        
        for word, phones in data:
            char_indices = [self.char_to_idx.get(c.lower(), self.char_to_idx['<unk>']) for c in word]
            char_indices = [self.char_to_idx['<sos>']] + char_indices + [self.char_to_idx['<eos>']]
            
            phone_indices = [self.phone_to_idx.get(p, self.phone_to_idx['<unk>']) for p in phones]
            phone_indices = [self.phone_to_idx['<sos>']] + phone_indices + [self.phone_to_idx['<eos>']]
            
            char_indices = self._pad_sequence(char_indices, self.max_len, self.char_to_idx['<pad>'])
            phone_indices = self._pad_sequence(phone_indices, self.max_len, self.phone_to_idx['<pad>'])
            
            converted.append((char_indices, phone_indices))
        
        return converted
    
    def _pad_sequence(self, seq: List[int], max_len: int, pad_token: int) -> List[int]:
        """Pad or truncate sequence to max_len"""
        if len(seq) > max_len:
            return seq[:max_len-1] + [seq[-1]] # Ensure <eos> is kept
        return seq + [pad_token] * (max_len - len(seq))
    
    def random_split(self, train_size: float = 0.8) -> Tuple['PhoneDataset', 'PhoneDataset']:
        """Randomly split dataset into training and validation sets"""
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        
        split_idx = int(len(self.data) * train_size)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_dataset = self._create_split_dataset([self.data[i] for i in train_indices])
        val_dataset = self._create_split_dataset([self.data[i] for i in val_indices])
        
        return train_dataset, val_dataset
    
    def _create_split_dataset(self, data_subset: List[Tuple[List[int], List[int]]]) -> 'PhoneDataset':
        """Helper to create a new dataset instance with a subset of data"""
        new_dataset = PhoneDataset.__new__(PhoneDataset)
        new_dataset.data = data_subset
        new_dataset.max_len = self.max_len
        
        # Copy vocabularies and settings
        new_dataset.char_to_idx = self.char_to_idx.copy()
        new_dataset.phone_to_idx = self.phone_to_idx.copy()
        new_dataset.idx_to_char = self.idx_to_char.copy()
        new_dataset.idx_to_phone = self.idx_to_phone.copy()
        new_dataset.direction = self.direction
        
        return new_dataset

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        char_seq, phone_seq = self.data[idx]
        if self.direction == 'g2p':
            return torch.tensor(char_seq, dtype=torch.long), torch.tensor(phone_seq, dtype=torch.long)
        else:  # p2g
            return torch.tensor(phone_seq, dtype=torch.long), torch.tensor(char_seq, dtype=torch.long)
    
    def set_direction(self, direction: str):
        """Set the dataset's direction ('g2p' or 'p2g')"""
        if direction not in ['g2p', 'p2g']:
            raise ValueError("Direction must be 'g2p' or 'p2g'")
        self.direction = direction