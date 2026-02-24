import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
import time
import os


# ============================================================
# Base Language Model Interface
# ============================================================

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


# ============================================================
# Uniform Baseline
# ============================================================

class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


# ============================================================
# Transformer Language Model
# ============================================================

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_index,
                 num_positions=20,
                 num_classess=27,
                 num_layers=1,
                 d_model=32,
                 nhead=4,
                 dim_feedforward=128,
                 dropout=0.1):

        super().__init__()

        self.vocab_index = vocab_index
        self.num_positions = num_positions
        self.num_classess = num_classess

        self.tok_embedding = nn.Embedding(num_classess, d_model)
        self.pos_embedding = nn.Embedding(num_positions, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classess)

    def causal_mask(self, N, device):
        return torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=device),
            diagonal=1
        )

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.long().to(device)

        Nmax = self.num_positions

        # Pad or truncate
        if x.numel() < Nmax:
            pad_id = self.vocab_index.index_of(' ')
            pad = torch.full(
                (Nmax - x.numel(),),
                pad_id,
                dtype=torch.long,
                device=device
            )
            x = torch.cat([x, pad], dim=0)
        elif x.numel() > Nmax:
            x = x[:Nmax]

        N = x.size(0)

        # Add batch dimension
        x = x.unsqueeze(0)

        pos_ids = torch.arange(N, device=device).unsqueeze(0)

        src = self.tok_embedding(x) + self.pos_embedding(pos_ids)

        causal = self.causal_mask(N, device)

        tgt = src

        memory = self.encoder(src, mask=causal)
        out = self.decoder(tgt, memory,
                           tgt_mask=causal,
                           memory_mask=causal)

        logits = self.classifier(out)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs.squeeze(0)


# ============================================================
# Neural LM Wrapper
# ============================================================

class NeuralLanguageModel(LanguageModel):
    def __init__(self, model: TransformerLM, vocab_index, num_positions=20):
        self.model = model
        self.vocab_index = vocab_index
        self.num_positions = num_positions
        self.model.eval()

    def _context_to_tensor(self, context: str):
        context = context[-self.num_positions:]
        if len(context) < self.num_positions:
            context = (" " * (self.num_positions - len(context))) + context

        idx = [self.vocab_index.index_of(c) for c in context]
        x = torch.tensor(idx, dtype=torch.long)

        device = next(self.model.parameters()).device
        return x.to(device)

    def get_next_char_log_probs(self, context):
        x = self._context_to_tensor(context)
        log_probs = self.model(x)
        last = log_probs[-1]
        return last.detach().cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        total = 0.0
        for ch in next_chars:
            lp = self.get_next_char_log_probs(context)
            total += float(lp[self.vocab_index.index_of(ch)])
            context = context + ch
        return total


# ============================================================
# Training Function (CPU / GPU / TPU Support)
# ============================================================

def train_lm(args, train_text, dev_text, vocab_index):

    model = TransformerLM(vocab_index)

    # TPU Detection
    if 'COLAB_TPU_ADDR' in os.environ:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        use_tpu = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_tpu = False

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_function = nn.NLLLoss()

    num_epochs = 1
    start = time.time()

    for epoch in range(num_epochs):
        loss_this_epoch = 0.0
        random.seed(epoch)

        for i in range(0, len(train_text) - 21):

            optimizer.zero_grad()

            x_str = train_text[i:i+20]
            x_idx = [vocab_index.index_of(c) for c in x_str]
            x_tensor = torch.tensor(x_idx, dtype=torch.long).to(device)

            y_str = train_text[i+1:i+21]
            y_idx = [vocab_index.index_of(c) for c in y_str]
            y_tensor = torch.tensor(y_idx, dtype=torch.long).to(device)

            log_probs = model(x_tensor)
            loss = loss_function(log_probs, y_tensor)

            loss.backward()

            if use_tpu:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()

            loss_this_epoch += loss.item()

        print(f"epoch {epoch}: loss={loss_this_epoch:.4f}")

    print(f"Total time: {time.time() - start:.3f} seconds")

    return NeuralLanguageModel(model, vocab_index)