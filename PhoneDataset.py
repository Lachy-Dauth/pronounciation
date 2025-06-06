import random
import pandas as pd
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import math

class PhoneDataset(Dataset):
    def __init__(self, data_path: str, freq_path: Optional[str] = None, max_words: int = 20000, 
                 max_len: int = 32):
        self.max_len = max_len
        self.data = []
        self.char_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.phone_to_idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.idx_to_char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.idx_to_phone = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.direction = 'g2p'  # Default direction is G2P
        self.word_weights = {}
        
        # Load frequency data if provided
        word_frequencies = {}
        if freq_path:
            word_frequencies = self._load_frequencies(freq_path, max_words)
        
        # Parse CMU dictionary
        raw_data = self._parse_cmu_dict(data_path)
        
        # Filter and sort by frequency if frequencies are available
        if word_frequencies:
            raw_data = self._filter_by_frequency(raw_data, word_frequencies, max_words)
        else:
            raw_data = raw_data[:max_words]  # Just take first max_words if no frequencies
        
        # Build vocabularies
        self._build_vocabularies(raw_data)
        
        # Convert to indices
        self.data = self._convert_to_indices(raw_data)
        
        print(f"Loaded {len(self.data)} word-pronunciation pairs")
        print(f"Character vocab size: {len(self.char_to_idx)}")
        print(f"Phone vocab size: {len(self.phone_to_idx)}")
    
    def _load_frequencies(self, freq_path: str, max_words: int) -> dict:
        """Load word frequencies from CSV file"""
        word_frequencies = {}
        try:
            freq_df = pd.read_csv(freq_path)
            count = 0
            for _, row in freq_df.iterrows():
                if count >= max_words:
                    break
                
                word = str(row['word']).upper() if pd.notna(row['word']) else "NULL"
                frequency = int(row['count']) if pd.notna(row['count']) else 1
                word_frequencies[word] = frequency
                count += 1
                
        except Exception as e:
            print(f"Warning: Could not load frequency file {freq_path}: {e}")
            
        return word_frequencies
    
    def _filter_by_frequency(self, raw_data: List[Tuple[str, List[str]]], 
                           word_frequencies: dict, max_words: int) -> List[Tuple[str, List[str]]]:
        """Filter and sort data by word frequency"""
        # Add frequency information and filter
        freq_filtered = []
        for word, phones in raw_data:
            if word in word_frequencies:
                freq_filtered.append((word, phones, word_frequencies[word]))
        
        # Sort by frequency (highest first)
        freq_filtered.sort(key=lambda x: x[2], reverse=True)
        
        # Create word weights for sampling
        for word, _, freq in freq_filtered[:max_words]:
            self.word_weights[word] = self._calculate_weight(freq)
        
        # Return just word-phone pairs
        return [(w, p) for w, p, _ in freq_filtered[:max_words]]
    
    def _calculate_weight(self, freq: int) -> float:
        """Calculate sampling weight from frequency"""
        return max(1.0, math.sqrt(freq))  # Square root to reduce extreme weights
    
    def _parse_cmu_dict(self, data_path: str) -> List[Tuple[str, List[str]]]:
        """Parse CMU pronunciation dictionary"""
        data = []
        with open(data_path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith(';;;'):
                    continue
                
                # Split on first space(s)
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                
                word, phones_str = parts
                
                # Skip alternative pronunciations (contains parentheses)
                if '(' in word:
                    continue
                
                # Skip words that don't start with A-Z
                if not word[0].isalpha():
                    continue
                
                # Clean up phones - separate stress markers
                phones = []
                for phone in phones_str.split():
                    if phone[-1].isdigit():
                        phones.append(phone[:-1])  # Phone without stress
                        phones.append(phone[-1])  # Stress marker
                    else:
                        phones.append(phone)
                    
                data.append((word.upper(), phones))
        
        return data
    
    def _build_vocabularies(self, data: List[Tuple[str, List[str]]]):
        """Build character and phone vocabularies"""
        chars = set()
        phones = set()
        
        for word, phone_seq in data:
            chars.update(word.lower())
            phones.update(phone_seq)
        
        # Add characters to vocab
        for char in sorted(chars):
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        
        # Add phones to vocab
        for phone in sorted(phones):
            if phone not in self.phone_to_idx:
                idx = len(self.phone_to_idx)
                self.phone_to_idx[phone] = idx
                self.idx_to_phone[idx] = phone
    
    def _convert_to_indices(self, data: List[Tuple[str, List[str]]]) -> List[Tuple[List[int], List[int]]]:
        """Convert words and phones to index sequences"""
        converted = []
        
        for word, phones in data:
            # Convert word to character indices
            char_indices = [self.char_to_idx.get(c.lower(), self.char_to_idx['<unk>']) 
                           for c in word]
            char_indices = [self.char_to_idx['<sos>']] + char_indices + [self.char_to_idx['<eos>']]
            
            # Convert phones to indices
            phone_indices = [self.phone_to_idx.get(p, self.phone_to_idx['<unk>']) 
                            for p in phones]
            phone_indices = [self.phone_to_idx['<sos>']] + phone_indices + [self.phone_to_idx['<eos>']]
            
            # Pad sequences
            char_indices = self._pad_sequence(char_indices, self.max_len, self.char_to_idx['<pad>'])
            phone_indices = self._pad_sequence(phone_indices, self.max_len, self.phone_to_idx['<pad>'])
            
            converted.append((char_indices, phone_indices))
        
        return converted
    
    def _pad_sequence(self, seq: List[int], max_len: int, pad_token: int) -> List[int]:
        """Pad sequence to max_len"""
        if len(seq) > max_len:
            return seq[:max_len]
        return seq + [pad_token] * (max_len - len(seq))
    
    def indices_to_word(self, indices: List[int]) -> str:
        """Convert character indices back to word"""
        chars = []
        for idx in indices:
            if idx in [0, 1, 2]:  # Skip pad, sos, eos
                continue
            if idx == 3:  # unk token
                chars.append('?')
            else:
                chars.append(self.idx_to_char.get(idx, '?'))
        return ''.join(chars).upper()
    
    def indices_to_phones(self, indices: List[int]) -> List[str]:
        """Convert phone indices back to phone sequence"""
        phones = []
        for idx in indices:
            if idx in [0, 1, 2]:  # Skip pad, sos, eos
                continue
            if idx == 3:  # unk token
                phones.append('<?>')
            else:
                phones.append(self.idx_to_phone.get(idx, '<?>'))
        return phones
    
    def random_split(self, train_size: float = 0.8) -> Tuple['PhoneDataset', 'PhoneDataset']:
        """Randomly split dataset into training and validation sets"""
        indices = list(range(len(self.data)))
        random.shuffle(indices)
        
        split_idx = int(len(self.data) * train_size)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Create new dataset instances
        train_dataset = self._create_split_dataset([self.data[i] for i in train_indices])
        val_dataset = self._create_split_dataset([self.data[i] for i in val_indices])
        
        return train_dataset, val_dataset
    
    def _create_split_dataset(self, data_subset: List[Tuple[List[int], List[int]]]) -> 'PhoneDataset':
        """Create a new dataset instance with a subset of data"""
        new_dataset = PhoneDataset.__new__(PhoneDataset)
        new_dataset.data = data_subset
        new_dataset.max_len = self.max_len
        
        # Copy vocabularies
        new_dataset.char_to_idx = self.char_to_idx.copy()
        new_dataset.phone_to_idx = self.phone_to_idx.copy()
        new_dataset.idx_to_char = self.idx_to_char.copy()
        new_dataset.idx_to_phone = self.idx_to_phone.copy()
        
        new_dataset.direction = self.direction
        
        # Copy relevant word weights
        new_dataset.word_weights = {}
        for char_seq, phone_seq in data_subset:
            if self.direction == 'g2p':
                word = self.indices_to_word(char_seq)
            else:
                word = self.indices_to_word(phone_seq)
            
            if word in self.word_weights:
                new_dataset.word_weights[word] = self.word_weights[word]
        
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
        """Set the direction for the dataset (g2p or p2g)"""
        if direction not in ['g2p', 'p2g']:
            raise ValueError("Direction must be 'g2p' or 'p2g'")
        self.direction = direction


class WeightedSampler:
    """Custom sampler that weights samples by word frequency"""
    def __init__(self, dataset: PhoneDataset, replacement: bool = True):
        self.dataset = dataset
        self.replacement = replacement
        self.weights = self._calculate_weights()
    
    def _calculate_weights(self) -> torch.Tensor:
        """Calculate sampling weights for each sample"""
        weights = []
        
        for char_seq, phone_seq in self.dataset.data:
            # Get the word based on current direction
            if self.dataset.direction == 'g2p':
                word = self.dataset.indices_to_word(char_seq)
            else:  # p2g
                word = self.dataset.indices_to_word(phone_seq)
            
            # Get weight for this word
            weight = self.dataset.word_weights.get(word, 1.0)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float)
    
    def __iter__(self):
        if self.replacement:
            # Sample with replacement
            for _ in range(len(self.dataset)):
                yield torch.multinomial(self.weights, 1).item()
        else:
            # Sample without replacement
            if len(self.weights) > 0:
                indices = torch.multinomial(self.weights, len(self.dataset), replacement=False)
                for idx in indices:
                    yield idx.item()
    
    def __len__(self):
        return len(self.dataset)