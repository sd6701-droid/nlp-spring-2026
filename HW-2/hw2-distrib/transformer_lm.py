# models.py

import numpy as np
import torch.nn as nn
from torch import optim
import random
import time
import torch.nn.functional as F
import torch

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")





# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)



class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

class TransformerLM(nn.Module):
    def __init__(self, vocab_index, num_positions = 20, num_classess = 27, num_layers = 1, d_model = 32, d_internal = 64,
                 nhead=4,
                 dim_feedforward=128,
                 dropout=0.1,):

        super().__init__()

        self.vocab_index = vocab_index
        self.num_positions = num_positions
        self.num_classess = num_classess
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_internal = d_internal

        #character embedding
        self.tok_embedding = nn.Embedding(27, d_model)

        #positional encoding 

        self.pos_embedding = nn.Embedding(num_positions, d_model)

        #create transformer layers
        encoder = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            batch_first = True
        )

        self.encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)

        decoder_layer =  nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = num_layers)

        #final classifier at the position
        self.classifier = nn.Linear(d_model, num_classess)

    def causal_mask(self, N, device):
        return torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x):

        device = next(self.parameters()).device
        x = x.long().to(device)

        # If single sample (20,) → make it (1, 20)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size, seq_len = x.shape

        Nmax = self.num_positions

        if seq_len < Nmax:
            pad_id = self.vocab_index.index_of(' ')
            pad = torch.full(
                (batch_size, Nmax - seq_len),
                pad_id,
                dtype=torch.long,
                device=device
            )
            x = torch.cat([x, pad], dim=1)
        elif seq_len > Nmax:
            x = x[:, :Nmax]

        batch_size, N = x.shape

        pos_ids = torch.arange(N, device=device).unsqueeze(0).expand(batch_size, N)

        src = self.tok_embedding(x) + self.pos_embedding(pos_ids)

        causal = self.causal_mask(N, device)

        memory = self.encoder(src, mask=causal)
        out = self.decoder(src, memory,
                        tgt_mask=causal,
                        memory_mask=causal)

        logits = self.classifier(out)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs

class NeuralLanguageModel(LanguageModel):
    def __init__(self, model: TransformerLM, vocab_index, num_positions: int = 20):
        self.model = model
        self.vocab_index = vocab_index
        self.num_positions = num_positions
        self.model.eval()

    def _context_to_tensor(self, context: str) -> torch.Tensor:
        # take last num_positions chars; left-pad with spaces if too short
        context = context[-self.num_positions:]
        if len(context) < self.num_positions:
            context = (" " * (self.num_positions - len(context))) + context

        idx = [self.vocab_index.index_of(c) for c in context]
        x = torch.tensor(idx, dtype=torch.long)

        # put on same device as model
        device = next(self.model.parameters()).device
        return x.to(device)
    
    def get_next_char_log_probs(self, context):
        x = self._context_to_tensor(context)          # (20,)
        log_probs = self.model(x)                     # (20, 27)
        log_probs = self.model(x)

        # If batched, take first batch
        if log_probs.dim() == 3:
            log_probs = log_probs[0]

        last = log_probs[-1]                # (27,)
        return last.detach().cpu().numpy()


    def get_log_prob_sequence(self, next_chars, context):
        total = 0.0
        for ch in next_chars:
            lp = self.get_next_char_log_probs(context)
            total += float(lp[self.vocab_index.index_of(ch)])
            context = context + ch
        return total



def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """


    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    model = TransformerLM(
                vocab_index, 
                num_positions = 20, 
                num_classess = 27,
                num_layers = 1,
                d_model = 32,
                d_internal = 64,
                nhead=4,
                dim_feedforward=128,
                dropout=0.1
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.zero_grad()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)

    num_epochs = 11
    start = time.time()

    for epoch in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(epoch)
        
        loss_function = nn.NLLLoss()
        batch_size = 128

        for i in range(0, len(train_text) - 1000, batch_size):
            batch_x = []
            batch_y = []

            for j in range(batch_size):
                x_str = train_text[i+j : i+j+20]
                y_str = train_text[i+j+1 : i+j+21]

                batch_x.append([vocab_index.index_of(c) for c in x_str])
                batch_y.append([vocab_index.index_of(c) for c in y_str])

            x_tensor = torch.tensor(batch_x).to(device)
            y_tensor = torch.tensor(batch_y).to(device)

            log_probs = model(x_tensor)
            loss = loss_function(log_probs.view(-1, 27), y_tensor.view(-1))
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()

        print(f"epoch {epoch}: loss={loss_this_epoch:.4f}")

    end = time.time()
    print(f"Total time: {end - start:.3f} seconds")

    return NeuralLanguageModel(model, vocab_index, num_positions=20)
