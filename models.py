# models.py

import math
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        print('Implement Unigram feature extractor')
        self.indexer = indexer
 
    def extract_features(self, sentence, add_to_indexer = False):
        feats = Counter()
        # print('sentence', sentence, add_to_indexer)
        for word in sentence:
            word = word.lower()
            feat = "Unigram=" + word

            if add_to_indexer: 
                fid = self.indexer.add_and_get_index(feat)
                feats[fid] += 1
            else: 
                fid = self.indexer.index_of(feat)
                if fid != -1:
                    feats[fid] += 1

        return feats


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    
    def __init__(self, indexer: Indexer):
        print('Implementing Bigram Feature extractor')
        self.indexer = indexer
        # raise Exception("Must be implemented")
    
    def extract_features(self, sentence, add_to_indexer = False):
        feats = Counter()
        for i in range(0, len(sentence) - 1):
            w = sentence[i]
            nw = sentence[i + 1]
            w = w.lower()
            nw = nw.lower()
            feat = "Bigram=" + w + ' ' + nw

            if add_to_indexer: 
                fid = self.indexer.add_and_get_index(feat)
                feats[fid] += 1
            else: 
                fid = self.indexer.index_of(feat)
                if fid != -1:
                    feats[fid] += 1

        return feats

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")



def _sigmoid(s: float) -> float:
    if s >= 0:
        z = math.exp(-s)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(s)
        return z / (1.0 + z)

def _sparse_dot(weights: np.ndarray, feats: Counter) -> float:
    return sum(weights[fid] * val for fid, val in feats.items())

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, weights: np.ndarray, bias: float, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.bias = bias
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        feats = self.feat_extractor.extract_features(sentence, add_to_indexer=False)
        score = self.bias + _sparse_dot(self.weights, feats)
        p = _sigmoid(score)
        return 1 if p >= 0.5 else 0

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    for ex in train_exs:
        # Adjust if your SentimentExample field name differs
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    # Try to locate the indexer inside your extractor (common names)
    indexer = getattr(feat_extractor, "indexer", None)
    if indexer is None:
        indexer = getattr(feat_extractor, "feat_indexer", None)
    if indexer is None:
        raise ValueError("Couldn't find indexer on feat_extractor (expected .indexer or .feat_indexer).")

    num_feats = len(indexer)


    weights = np.zeros(num_feats, dtype=np.float32)
    bias = 0.0


    epochs = 35
    lr = 0.1


    l2 = 0.0     
    seed = 0
    rng = random.Random(seed)


    for ep in range(epochs):
        rng.shuffle(train_exs)

        for ex in train_exs:
            words = ex.words
            y = ex.label  # 0/1

            feats = feat_extractor.extract_features(words, add_to_indexer=False)

            score = bias + _sparse_dot(weights, feats)
            # print('score', score)
            p = _sigmoid(score)

            # error term
            err = p - y

            # lr_t = lr                 # constant
            # lr_t = lr * (0.25 ** ep)  # decay
            lr_t = (lr + 1.225) / (1 + ep)     # 1/t (78% dev acc)

            for fid, val in feats.items():
                grad = err * val
                if l2 != 0.0:
                    grad += l2 * weights[fid]
                weights[fid] -= lr_t * grad

            bias -= lr_t * err

    return LogisticRegressionClassifier(weights, bias, feat_extractor)

def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network, word_embeddings):
        self.network = network
        self.word_embeddings = word_embeddings

    def predict(self, ex_words):
        self.network.eval()
        with torch.no_grad():

            vecs = [self.word_embeddings.get_embedding(w) for w in ex_words]
            x_tensor = torch.from_numpy(np.stack(vecs, axis=0)).float()  # (num_words, 300)
            x_avg = x_tensor.mean(dim=0)                                  # (300,)

            log_probs = self.network(x_avg)
        return int(torch.argmax(log_probs).item())


class FFNN(nn.Module):

    def __init__(self, inp, hid1, hid2, out, pdrop=0.4):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        # super(FFNN, self).__init__()
        # self.V = nn.Linear(inp, hid)
        # self.g = nn.ReLU()
        # self.W = nn.Linear(hid, out)
        # self.log_softmax = nn.LogSoftmax(dim=0)
        # #intialize weights according to a formula Xavier Glorot.
        # # nn.init.xavier_uniform_(self.V.weight)
        # # nn.init.xavier_uniform_(self.W.weight)
        # #initialize weights with zeros 
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)


        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(inp, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, out)
        self.act = nn.ReLU()
        # self.drop = nn.Dropout(pdrop)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        



    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """

        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        logits = self.fc3(h2)
        return self.log_softmax(logits)



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    # for exs in train_exs:
    #     for word in exs.words:
    #         # get the word embedding of each world
    #         print(word_embeddings.get_embedding(word).shape)

        

    feat_vec_size = 300
    # Let's use 4 hidden units
    embedding_size = 400

    num_classes = 2

    num_epochs = 15

    ffnn = FFNN(feat_vec_size, 400 , 500, num_classes)

    initial_learning_rate = 0.01
    
    optimizer = optim.Adam(ffnn.parameters(), lr = initial_learning_rate)

    loss_fn = torch.nn.NLLLoss()

    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0

        for idx in ex_indices:

            x = [word_embeddings.get_embedding(word) for word in train_exs[idx].words]

            x_tensor = torch.from_numpy(np.stack(x, axis=0)).float()  
            x_avg = x_tensor.mean(dim=0)                              


            y = train_exs[idx].label

            # y_onehot = torch.zeros(num_classes)
            # y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)

            ffnn.zero_grad()

            log_probs = ffnn(x_avg)

            # y is an int in {0,1}
            y_t = torch.tensor([y], dtype=torch.long)
            
            

            log_probs = ffnn(x_avg)                 # shape: [2]
            loss = loss_fn(log_probs.unsqueeze(0), y_t)

            total_loss += loss

            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    
    #evaluate on the train set
    train_correct = 0
    for idx in range(0, len(train_exs)):
        x = [word_embeddings.get_embedding(word) for word in train_exs[idx].words]

        x_tensor = torch.from_numpy(np.stack(x, axis=0)).float()  # (num_words, 300)
        x_avg = x_tensor.mean(dim=0)  

        y = train_exs[idx].label
        log_probs = ffnn.forward(x_avg)
        prediction = torch.argmax(log_probs)
        if y == prediction:
            train_correct += 1
        # print("Example " +  + "; gold = " + repr(train_exs[idx].label) + "; pred = " +\
        #       repr(prediction) + " with probs " + repr(log_probs))
    print(f"{train_correct}/{len(train_exs)} correct after training")
    print(f"Train accuracy: {train_correct/len(train_exs):.4f}")


    # print("Implement Nerual Network Model", dev_exs)
    print("Implement Nerual Network Model", word_embeddings)
    return NeuralSentimentClassifier(ffnn, word_embeddings)



def train_batch_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    print("Implement Nerual Network Model", train_exs[0])

    # for exs in train_exs:
    #     for word in exs.words:
    #         # get the word embedding of each world
    #         print(word_embeddings.get_embedding(word).shape)

        
    #
    # Define some constants

    feat_vec_size = 300

    embedding_size = 400
   
    num_classes = 2

    num_epochs = 150

    batch_size = 64

    ffnn = FFNN(feat_vec_size, 400 , 500, num_classes)

    initial_learning_rate = 0.01
    optimizer = optim.Adam(ffnn.parameters(), lr = initial_learning_rate)

    criterion = nn.NLLLoss()

    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0

        ffnn.train()

        for idx in range (0, len(ex_indices), batch_size):

            batch_ids = ex_indices[idx:idx + batch_size]
            
            batch_exs = [train_exs[i] for i in batch_ids]

            X_np = np.stack([
                np.mean(np.stack([word_embeddings.get_embedding(w) for w in ex.words], axis=0), axis=0)
                for ex in batch_exs
            ], axis=0)                         # (N, 300)

            X = torch.from_numpy(X_np).float()                        


            # y = train_exs[idx].label

            # y_onehot = torch.zeros(num_classes)
            # y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)

            y = torch.tensor([ex.label for ex in batch_exs], dtype=torch.long)

            ffnn.zero_grad()

            log_probs = ffnn.forward(X)
            loss = criterion(log_probs, y)
            total_loss += loss

            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    
    #evaluate on the train set
    train_correct = 0
    for idx in range(0, len(train_exs)):
        x = [word_embeddings.get_embedding(word) for word in train_exs[idx].words]

        x_tensor = torch.from_numpy(np.stack(x, axis=0)).float()  # (num_words, 300)
        x_avg = x_tensor.mean(dim=0)  

        y = train_exs[idx].label
        log_probs = ffnn.forward(x_avg)
        prediction = torch.argmax(log_probs)
        if y == prediction:
            train_correct += 1
        # print("Example " +  + "; gold = " + repr(train_exs[idx].label) + "; pred = " +\
        #       repr(prediction) + " with probs " + repr(log_probs))
    print(f"{train_correct}/{len(train_exs)} correct after training")
    print(f"Train accuracy: {train_correct/len(train_exs):.4f}")


    # print("Implement Nerual Network Model", dev_exs)
    print("Implement Nerual Network Model", word_embeddings)
    return NeuralSentimentClassifier(ffnn, word_embeddings)
    # raise NotImplementedError



#batched dan


# class BatchedNeuralSentimentClassifier(SentimentClassifier):
#     def __init__(self, network, word_embeddings, batch_size=64):
#         self.network = network
#         self.word_embeddings = word_embeddings
#         self.batch_size = batch_size

#         idxer = word_embeddings.word_indexer
#         self.pad_idx = idxer.index_of("PAD")  # 0
#         self.unk_idx = idxer.index_of("UNK")  # 1

#     def _words_to_ids(self, words: List[str]) -> List[int]:
#         idxer = self.word_embeddings.word_indexer
#         ids = []
#         for w in words:
#             wid = idxer.index_of(w)
#             ids.append(wid if wid != -1 else self.unk_idx)
#         return ids

#     def _make_padded_X(self, batch_words: List[List[str]]) -> torch.Tensor:
#         # batch_words is list of sentences (list of tokens)
#         seqs = [self._words_to_ids(words) for words in batch_words]
#         lengths = [len(s) for s in seqs]
#         T = max(lengths) if len(seqs) > 0 else 1
#         B = len(seqs)

#         X_idx = torch.full((B, T), self.pad_idx, dtype=torch.long)
#         for i, s in enumerate(seqs):
#             X_idx[i, :len(s)] = torch.tensor(s, dtype=torch.long)

#         return X_idx  # (B, T) LongTensor

#     def predict(self, ex_words: List[str]) -> int:
#         self.network.eval()
#         with torch.no_grad():
#             X_idx = self._make_padded_X([ex_words])   # (1, T) Long
#             log_probs = self.network(X_idx)          # (1, 2)
#             return int(torch.argmax(log_probs, dim=1).item())

#     def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
#         self.network.eval()
#         preds = []
#         with torch.no_grad():
#             for start in range(0, len(all_ex_words), self.batch_size):
#                 batch_words = all_ex_words[start:start + self.batch_size]
#                 X_idx = self._make_padded_X(batch_words)         # (B, T)
#                 log_probs = self.network(X_idx)                  # (B, 2)
#                 batch_preds = torch.argmax(log_probs, dim=1)      # (B,)
#                 preds.extend(batch_preds.tolist())
#         return preds

# class BatchedDAN(nn.Module):

#     def __init__(self, word_embeddings, hid1=200, hid2=200, num_classes=2, frozen=False):
#         super().__init__()

#         W = torch.tensor(word_embeddings.vectors, dtype=torch.float32)  # (vocab, 300)

#         # PAD is index 0 in your indexer; this tells PyTorch to treat row 0 as padding
#         self.emb = nn.Embedding.from_pretrained(W, freeze=frozen, padding_idx=0)

#         emb_dim = W.size(1)  # 300
#         self.fc1 = nn.Linear(emb_dim, hid1)
#         self.fc2 = nn.Linear(hid1, hid2)
#         self.fc3 = nn.Linear(hid2, num_classes)
#         self.act = nn.ReLU()
#         self.log_softmax = nn.LogSoftmax(dim=-1)

#     def forward(self, X_idx):
#         # X_idx: (B, T) padded with 0
#         E = self.emb(X_idx)  # (B, T, 300)

#         # mask: 1 for real tokens, 0 for PAD (PAD id = 0)
#         mask = (X_idx != 0).float()              # (B, T)
#         lengths = mask.sum(dim=1).clamp(min=1.0) # (B,)

#         # masked sum: PAD positions contribute 0
#         summed = (E * mask.unsqueeze(-1)).sum(dim=1)  # (B, 300)

#         # average only over real tokens
#         avg = summed / lengths.unsqueeze(1)           # (B, 300)

#         self.drop = nn.Dropout(0.3)


#         h1 = self.act(self.fc1(avg))
#         h2 = self.act(self.fc2(h1))
#         logits = self.fc3(h2)
#         return self.log_softmax(logits)



# def words_to_ids(words, word_embeddings):
#     idxer = word_embeddings.word_indexer
#     unk = idxer.index_of("UNK")
#     ids = []
#     for w in words: 
#         wid = idxer.index_of(w)
#         ids.append(wid if wid != -1 else unk)
#     return ids

# def make_padded_batch(batch_exs, word_embeddings):
#     idxer = word_embeddings.word_indexer
#     pad = idxer.index_of("PAD")
    
#     seqs = [words_to_ids(ex.words, word_embeddings) for ex in batch_exs]
#     lengths = torch.tensor([len(s) for s in seqs])
#     T = int(lengths.max().item())
#     B = len(seqs)

#     X = torch.full((B, T), pad, dtype=torch.long)


#     for i, s in enumerate(seqs):
#        X[i, :len(s)] = torch.tensor(s, dtype = torch.long)
    
#     y = torch.tensor([ex.label for ex in batch_exs], dtype=torch.long)

#     return X, lengths, y




# def train_batched_dan(train_exs, word_embeddings, epochs=20, batch_size=32, lr=1e-3):
#     model = BatchedDAN(word_embeddings)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#     criterion = nn.NLLLoss()  # averages over batch by default

#     for ep in range(epochs):
#         indices = list(range(len(train_exs)))
#         random.shuffle(indices)
#         total_loss = 0.0
#         model.train()

#         for start in range(0, len(indices), batch_size):
#             batch_ids = indices[start:start+batch_size]
#             batch_exs = [train_exs[i] for i in batch_ids]

#             X_idx, lengths, y = make_padded_batch(batch_exs, word_embeddings)

#             optimizer.zero_grad()
#             log_probs = model(X_idx)   # (B, 2)
#             loss = criterion(log_probs, y)      # scalar
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"epoch {ep}: loss={total_loss:.4f}")

#     return BatchedNeuralSentimentClassifier(model, word_embeddings)
 




