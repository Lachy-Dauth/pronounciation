from typing import List
import torch
import torch.nn as nn
import math
import time
import torch.optim as optim
from torch.amp import autocast, GradScaler

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x shape: (batch, seq, d_model)
        return x + self.pe[:, :x.size(1), :]

    
class GenericTransformer(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                    d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                    dim_feedforward: int = 512, max_len: int = 32):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Embeddings
        self.input_embedding = nn.Embedding(input_size, d_model)
        self.output_embedding = nn.Embedding(output_size, d_model)
        
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
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        
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
        # src: (batch_size, seq_len)
        # tgt: (batch_size, seq_len)
        
        _, tgt_len = tgt.shape
        
        # Embeddings and positional encoding
        src_emb = self.input_embedding(src) * math.sqrt(self.d_model) # (batch, seq, d_model)
        tgt_emb = self.output_embedding(tgt) * math.sqrt(self.d_model) # (batch, seq, d_model)
        
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
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Output projection
        output = self.output_projection(output)  # (seq_len, batch, vocab_size)
        
        return output
    
class P2GTransformer(GenericTransformer):
    def __init__(self, phone_vocab_size: int, char_vocab_size: int, 
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4, 
                 dim_feedforward: int = 512, max_len: int = 32):
        super().__init__(
            input_size=phone_vocab_size,
            output_size=char_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len
        )

class G2PTransformer(GenericTransformer):
    def __init__(self, char_vocab_size: int, phone_vocab_size: int, 
                    d_model: int = 256, nhead: int = 8, num_layers: int = 4, 
                    dim_feedforward: int = 512, max_len: int = 32):
        super().__init__(
            input_size=char_vocab_size,
            output_size=phone_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len
        )

def train_model(model, train_loader, val_loader, device, num_epochs=50, language='English', learning_rate=0.005):
    """Training loop with validation"""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=learning_rate)
    
    best_val_loss = float('inf')

    model_name = model.__class__.__name__[0:3].lower()
    
    scaler = GradScaler('cuda')  # for AMP
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if batch_idx % max((len(train_loader) // 100), 1) == 0:
                elapsed = time.time() - start_time
                bps = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                instances = (batch_idx + 1) * train_loader.batch_size
                ips = instances / elapsed if elapsed > 0 else 0
                percent = (batch_idx + 1) / len(train_loader) * 100
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Batches/sec: {bps:.2f}, Instances/sec: {ips:.2f}, Progress: {percent:.2f}% {" " * 10}\r', end='')

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
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} {" " * 100}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'models/{language.lower()}_{model_name}_best_model.pth')
            print(f'Saved new best {model_name.upper()} model for {language}!')

def inference(model, dataset, phones_or_word, device, max_len=32):
    """Run inference on the model for a given input"""

    direction = dataset.direction
    model.eval()
    is_p2g = (direction == 'p2g')
    if is_p2g:
        # phones -> word
        indices = [dataset.phone_to_idx.get(p, dataset.phone_to_idx['<unk>']) for p in phones_or_word]
        sos = dataset.phone_to_idx['<sos>']
        eos = dataset.phone_to_idx['<eos>']
        vocab = dataset.char_to_idx
        idx_to_vocab = dataset.idx_to_char
    else:
        # word -> phones
        indices = [dataset.char_to_idx.get(c, dataset.char_to_idx['<unk>']) for c in phones_or_word]
        sos = dataset.char_to_idx['<sos>']
        eos = dataset.char_to_idx['<eos>']
        vocab = dataset.phone_to_idx
        idx_to_vocab = dataset.idx_to_phone

    indices = [sos] + indices + [eos]
    indices = dataset._pad_sequence(indices, max_len, 0)
    src = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
    tgt = torch.full((1, 1), vocab['<sos>'], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            output = model(src, tgt)
            next_token = output[0, -1, :].argmax().item()
            if next_token == vocab['<eos>']:
                break
            # expand tgt tensor in CUDA
            next_token_tensor = torch.full((1, 1), next_token, dtype=torch.long, device=device)
            tgt = torch.cat([tgt, next_token_tensor], dim=1)
    # Convert indices back to vocab
    idxs = tgt[0, 1:].cpu().numpy()
    out = [idx_to_vocab[idx] for idx in idxs if idx not in [0, 2]]  # 0=pad, 2=eos
    return out
