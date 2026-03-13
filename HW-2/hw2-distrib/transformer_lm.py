import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
import time


class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_positions=20, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        
        self.num_positions = num_positions
        self.d_model = d_model
        
        # Embeddings
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(num_positions, d_model)
        
        # Transformer encoder with causal masking (acts like a decoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def _generate_causal_mask(self, seq_len, device):
        """Creates upper triangular mask (True = masked/ignored)"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.bool()
        return mask
    
    def forward(self, x):
        """
        x: (batch_size, seq_len) tensor of token indices
        returns: (batch_size, seq_len, vocab_size) log probabilities
        """
        device = x.device
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        x = self.tok_embedding(x) + self.pos_embedding(positions)
        
        # Create causal mask
        causal_mask = self._generate_causal_mask(seq_len, device)
        
        # Pass through transformer
        x = self.transformer(x, mask=causal_mask)
        
        # Project to vocabulary
        logits = self.output_layer(x)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model: TransformerLM, vocab_index, num_positions=20):
        self.model = model
        self.vocab_index = vocab_index
        self.num_positions = num_positions
        self.device = next(model.parameters()).device
        self.model.eval()
    
    def _context_to_indices(self, context: str):
        """Convert context string to tensor of indices"""
        # Use last (num_positions - 1) chars of context, prepend space as start token
        context = context[-(self.num_positions - 1):]
        # Prepend space (start token)
        context = " " + context
        # Pad with spaces if needed
        if len(context) < self.num_positions:
            context = " " * (self.num_positions - len(context)) + context
        
        indices = [self.vocab_index.index_of(c) for c in context]
        return torch.tensor(indices, dtype=torch.long, device=self.device).unsqueeze(0)
    
    def get_next_char_log_probs(self, context) -> np.ndarray:
        """Returns log probs for next character given context"""
        with torch.no_grad():
            x = self._context_to_indices(context)
            log_probs = self.model(x)  # (1, seq_len, vocab_size)
            # Get prediction from last position
            return log_probs[0, -1, :].cpu().numpy()
    
    def get_log_prob_sequence(self, next_chars, context) -> float:
        """Score a sequence of characters following context"""
        total_log_prob = 0.0
        for ch in next_chars:
            log_probs = self.get_next_char_log_probs(context)
            char_idx = self.vocab_index.index_of(ch)
            total_log_prob += log_probs[char_idx]
            context = context + ch
        return total_log_prob


def train_lm(args, train_text, dev_text, vocab_index):
    """Train the language model"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_positions = 20
    d_model = 64
    nhead = 4
    num_layers = 3
    dim_feedforward = 256
    dropout = 0.1
    batch_size = 1
    learning_rate = 1e-3
    num_epochs = 10
    
    # Create model
    model = TransformerLM(
        vocab_size=27,
        num_positions=num_positions,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()
    
    # Prepare training data as chunks
    # Input: chars 0..19, Target: chars 1..20 (shifted by 1)
    chunk_size = num_positions
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Create chunks from training text
        # We need input (with space prepended) and target (the actual chars)
        indices = list(range(0, len(train_text) - chunk_size, chunk_size))
        random.shuffle(indices)
        
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            
            batch_x = []
            batch_y = []
            
            for idx in batch_indices:
                # Input: space + first 19 chars (positions 0-18 of chunk)
                # This predicts chars at positions 0-19 of chunk
                chunk = train_text[idx:idx + chunk_size]
                
                input_str = " " + chunk[:-1]  # space + first 19 chars
                target_str = chunk  # all 20 chars
                
                x_indices = [vocab_index.index_of(c) for c in input_str]
                y_indices = [vocab_index.index_of(c) for c in target_str]
                
                batch_x.append(x_indices)
                batch_y.append(y_indices)
            
            x_tensor = torch.tensor(batch_x, dtype=torch.long, device=device)
            y_tensor = torch.tensor(batch_y, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            log_probs = model(x_tensor)  # (batch, seq_len, vocab_size)
            
            # Reshape for loss computation
            loss = loss_fn(log_probs.view(-1, 27), y_tensor.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    
    return NeuralLanguageModel(model, vocab_index, num_positions)