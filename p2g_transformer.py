import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
import math
import random
from typing import List, Tuple, Dict

from PhoneDataset import PhoneDataset as Dataset, WeightedSampler

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class P2GTransformer(nn.Module):
    def __init__(self, phone_vocab_size: int, char_vocab_size: int, 
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4, 
                 dim_feedforward: int = 512, max_len: int = 32):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Embeddings (reversed from G2P)
        self.phone_embedding = nn.Embedding(phone_vocab_size, d_model)
        self.char_embedding = nn.Embedding(char_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=False
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, char_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq, pad_idx=0):
        """Create padding mask for attention"""
        return (seq == pad_idx)
    
    def create_causal_mask(self, size):
        """Create causal mask for decoder"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()
    
    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        # src: phones (batch_size, seq_len)
        # tgt: characters (batch_size, seq_len)
        
        batch_size, src_len = src.shape
        _, tgt_len = tgt.shape
        
        # Embeddings and positional encoding
        src_emb = self.phone_embedding(src) * math.sqrt(self.d_model)  # (batch, seq, d_model)
        tgt_emb = self.char_embedding(tgt) * math.sqrt(self.d_model)
        
        # Transpose for transformer (seq_len, batch, d_model)
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        
        # Add positional encoding
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Create masks
        if src_padding_mask is None:
            src_padding_mask = self.create_padding_mask(src)
        if tgt_padding_mask is None:
            tgt_padding_mask = self.create_padding_mask(tgt)
        
        tgt_mask = self.create_causal_mask(tgt_len).to(src.device)
        
        # Transformer forward pass
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=None,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Output projection
        output = self.output_projection(output)  # (seq_len, batch, vocab_size)
        output = output.transpose(0, 1)  # (batch, seq_len, vocab_size)
        
        return output

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    """Training loop with validation"""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # Teacher forcing: use target input (without last token) and target output (without first token)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            optimizer.zero_grad()
            
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_p2g_model.pth')
            print('Saved new best P2G model!')

def inference(model, dataset, phones: List[str], device, max_len=32):
    """Generate word for given phonemes"""
    model.eval()
    
    # Convert phones to indices
    phone_indices = [dataset.phone_to_idx.get(p, dataset.phone_to_idx['<unk>']) 
                    for p in phones]
    phone_indices = [dataset.phone_to_idx['<sos>']] + phone_indices + [dataset.phone_to_idx['<eos>']]
    phone_indices = dataset._pad_sequence(phone_indices, max_len, 0)
    
    src = torch.tensor(phone_indices).unsqueeze(0).to(device)  # (1, seq_len)
    
    # Start with SOS token
    tgt = torch.tensor([[dataset.char_to_idx['<sos>']]]).to(device)
    
    with torch.no_grad():
        for _ in range(max_len - 1):
            output = model(src, tgt)
            next_token = output[0, -1, :].argmax().item()
            
            if next_token == dataset.char_to_idx['<eos>']:
                break
            
            tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)
    
    # Convert back to characters
    char_indices = tgt[0][1:].cpu().numpy()  # Skip SOS token
    chars = [dataset.idx_to_char[idx] for idx in char_indices 
             if idx not in [0, 2]]  # Skip pad and EOS
    
    return ''.join(chars)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset
    print("Loading P2G dataset...")
    dataset = Dataset(
        data_path='cmudict.dict',  # Path to your CMU dictionary file
        freq_path='word_frequencies.csv',  # Path to your frequency file
        max_words=200000,
        max_len=32
    )
    
    dataset.set_direction('p2g')  # Set to P2G mode

    # Split dataset
    train_dataset, val_dataset = dataset.random_split(train_size=0.8)
    
    # Create data loaders with weighted sampling for training
    train_sampler = WeightedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        sampler=train_sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = P2GTransformer(
        phone_vocab_size=len(dataset.phone_to_idx),
        char_vocab_size=len(dataset.char_to_idx),
        d_model=128,
        nhead=16,
        num_layers=4,
        dim_feedforward=256,
        max_len=32
    ).to(device)
    
    print(f"P2G Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("Starting P2G training...")
    train_model(model, train_loader, val_loader, device, num_epochs=20)
    
    # Test inference
    print("\nTesting P2G inference...")
    test_phones = [
        ['HH', 'EH', '0', 'L', 'OW', '1'],  # hello
        ['W', 'ER', '1', 'L', 'D'],    # world
        ['K', 'AE', '1', 'T'],         # cat
        ['D', 'AO', '1', 'G']          # dog
    ]
    for phones in test_phones:
        word = inference(model, dataset, phones, device)
        print(f"{' '.join(phones)} -> {word}")

if __name__ == "__main__":
    main()