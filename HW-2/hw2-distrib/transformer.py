# transformer.py

import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import math

# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model =d_model
        self.d_internal = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers
        # raise Exception("Implement me")

        # Character embedding
        self.tok_embed = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learned, provided class below)
        self.pos_enc = PositionalEncoding(d_model=d_model, num_positions=num_positions, batched=False)

        # Stack of TransformerLayers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_internal=d_internal) for _ in range(num_layers)
        ])

        # Final classifier per position
        self.classifier = nn.Linear(d_model, num_classes)

        # Attention mode: "bidir" (default), "causal", or "anti_causal"
        # You can set this in train_classifier based on args.task if needed.
        self.attn_mode = "bidir"

    def _build_attn_mask(self, N: int, device: torch.device):
        """
        Returns a boolean mask of shape (N, N) where True means "blocked".
        """
        if self.attn_mode == "causal":
            # block future (j > i)
            return torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
        elif self.attn_mode == "anti_causal":
            # block past (j < i)
            return torch.tril(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=-1)
        else:
            return None
        


    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """

        # Accept a LetterCountingExample by mistake and extract tensor
        if hasattr(indices, "input_tensor"):
            indices = indices.input_tensor

        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)

        indices = indices.long()
        device = indices.device

        # Ensure length == num_positions (20) just in case
        N = self.num_positions
        if indices.numel() < N:
            pad_id = self.vocab_size - 1  # assumes space is last; OK for your vocab list
            pad = torch.full((N - indices.numel(),), pad_id, dtype=torch.long, device=device)
            indices = torch.cat([indices, pad], dim=0)
        elif indices.numel() > N:
            indices = indices[:N]

        # Embeddings: (20, d_model)
        x = self.tok_embed(indices)          # (N, D)
        # print('x-embedd ed toke', x)
        # x = self.pos_enc(x)                 # (N, D)
    
        attn_mask = self._build_attn_mask(N, device=device)

        attn_maps = []
        for layer in self.layers:
            # print('layer', layer)
            x, attn = layer(x, attn_mask=attn_mask)   # x: (N,D), attn:(N,N)
            attn_maps.append(attn)

        logits = self.classifier(x)                 # (N, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)   # (N, num_classes)

        return log_probs, attn_maps



        # raise Exception("Implement me")


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """

        # we need to write a single attnetion layer over here than extend it to multiple layers using
        # using transformer architecture
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal

        # Q/K/V projections: d_model -> d_internal
        self.Wq = nn.Linear(d_model, d_internal, bias=False)
        self.Wk = nn.Linear(d_model, d_internal, bias=False)
        self.Wv = nn.Linear(d_model, d_internal, bias=False)

        # Output projection back to d_model for residual
        self.Wo = nn.Linear(d_internal, d_model, bias=False)

        # LayerNorm + FFN (typical encoder-layer parts)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(4 * d_model, d_model)

    def forward(self, input_vecs, attn_mask=None):
        x = input_vecs  # (N, D)

        # ---- Self-attention  ----
        x_norm = x

        Q = self.Wq(x_norm)   # (N, d_internal)
        K = self.Wk(x_norm)   # (N, d_internal)
        V = self.Wv(x_norm)   # (N, d_internal)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_internal)  # (N, N)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)            # (N, N)
        context = attn @ V                          # (N, d_internal)
        attn_out = self.Wo(context)                 # (N, d_model)

        x = x + attn_out                            # residual

        # ---- Feed-forward (pre-LN) ----
        x_norm2 = x
         # self.ln2(x)
        ff = self.ff2(F.gelu(self.ff1(x_norm2)))    # (N, d_model)
        x = x + ff                                  # residual

        return x, attn
        # raise Exception("Implement me")


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


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # raise Exception("Not fully implemented yet")

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:

    model = Transformer(vocab_size= 27, num_positions= 20, num_classes= 3, num_layers= 2, d_model= 32, d_internal= 64)

    task = getattr(args, "task", "")
    print(task)
    if task == "BEFORE":
        model.attn_mode = "causal"
    elif task == "AFTER":
        model.attn_mode = "anti_causal"
    else:
        model.attn_mode = "bidir"


    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    start = time.time()

    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t) 
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()

        for ex_idx in ex_idxs:
            
            optimizer.zero_grad()
            log_probability, attn = model(train[ex_idx].input_tensor)
            # print('ouput from the model', log_probability, attn)

            actual_ouput = train[ex_idx].output_tensor.long()

            loss = loss_fcn(log_probability, actual_ouput) # TODO: Run forward and compute loss
            # model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print(f"epoch {t}: loss={loss_this_epoch:.4f}")
        end = time.time()
    print(f"Total time: {end - start:.3f} seconds")

    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
