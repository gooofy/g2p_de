#!/bin/env python3

# adapted to german mfa from: https://github.com/Kyubyong/nlp_made_easy/blob/master/PyTorch%20seq2seq%20template%20based%20on%20the%20g2p%20task.ipynb

import numpy as np
from tqdm import tqdm # from tqdm import tqdm_notebook as tqdm
from distance import levenshtein
from pathlib import Path
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from model import Encoder, Decoder, Net, Hparams, load_vocab
from g2pdata import G2pDataset, convert_ids_to_phonemes

LEXICON_DE = Path('/home/guenter/projects/hal9000/ennis/chat/src/efficientspeech/lexicon/german_mfa.dict')

MODEL_PATH = Path('g2p_de.ckpt')

# DEBUG_LIMIT = 23
DEBUG_LIMIT = 0

hp = Hparams()

g2idx, idx2g, p2idx, idx2p = load_vocab(hp)

# print (f"g2idx={g2idx}")
# print (f"p2idx={p2idx}")

def load_lex(lexicon_path:Path, hp:Hparams):

    lexicon = {} # word -> [ phonemes ]

    with open (lexicon_path, 'r') as lexf:
        for line in lexf:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            graph = parts[0]
            phonemes = parts[1].split(' ')
            valid = True
            for c in graph:
                if not c in hp.graphemes:
                    valid = False
                    break

            if not valid:
                continue

            # print (f"{graph} : {phonemes}")

            lexicon[graph] = phonemes

            if DEBUG_LIMIT and len(lexicon) >= DEBUG_LIMIT:
                break

    return lexicon


lexicon = load_lex(LEXICON_DE, hp)

def prepare_data(lexicon):

    words = []
    prons = []

    for w, p in lexicon.items():
        words.append(" ".join(list(w)))
        prons.append(" ".join(p))

    indices = list(range(len(words)))

    from random import shuffle
    shuffle(indices)
    words = [words[idx] for idx in indices]
    prons = [prons[idx] for idx in indices]
    num_train, num_test = int(len(words)*.8), int(len(words)*.1)
    train_words, eval_words, test_words = words[:num_train], \
                                          words[num_train:-num_test],\
                                          words[-num_test:]
    train_prons, eval_prons, test_prons = prons[:num_train], \
                                          prons[num_train:-num_test],\
                                          prons[-num_test:]
    return train_words, eval_words, test_words, train_prons, eval_prons, test_prons

train_words, eval_words, test_words, train_prons, eval_prons, test_prons = prepare_data(lexicon)
#print(train_words[0])
#print(train_prons[0])

def drop_lengthy_samples(words, prons, enc_maxlen, dec_maxlen):
    """We only include such samples less than maxlen."""
    _words, _prons = [], []
    for w, p in zip(words, prons):
        if len(w.split()) + 1 > enc_maxlen: continue
        if len(p.split()) + 1 > dec_maxlen: continue # 1: <EOS>
        _words.append(w)
        _prons.append(p)
    return _words, _prons

train_words, train_prons = drop_lengthy_samples(train_words, train_prons, hp.enc_maxlen, hp.dec_maxlen)
# We do NOT apply this constraint to eval and test datasets.

def pad(batch):
    '''Pads zeros such that the length of all samples in a batch is the same.'''
    f = lambda x: [sample[x] for sample in batch]
    x_seqlens = f(1)
    y_seqlens = f(5)
    words = f(2)
    prons = f(-1)

    x_maxlen = np.array(x_seqlens).max()
    y_maxlen = np.array(y_seqlens).max()

    f = lambda x, maxlen, batch: [sample[x]+[0]*(maxlen-len(sample[x])) for sample in batch]
    x = f(0, x_maxlen, batch)
    decoder_inputs = f(3, y_maxlen, batch)
    y = f(4, y_maxlen, batch)

    f = torch.LongTensor
    return f(x), x_seqlens, words, f(decoder_inputs), f(y), y_seqlens, prons

#
# Train & Eval functions
#

def train(model, iterator, optimizer, criterion, device):
    model.train()
    for i, batch in enumerate(iterator):
        x, x_seqlens, words, decoder_inputs, y, y_seqlens, prons = batch

        x, decoder_inputs = x.to(device), decoder_inputs.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, y_hat = model(x, x_seqlens, decoder_inputs)

        # calc loss
        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1) # (N*T,)
        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i and i%100==0:
            print(f"step: {i}, loss: {loss.item()}")

def calc_per(Y_true, Y_pred):
    '''Calc phoneme error rate
    Y_true: list of predicted phoneme sequences. e.g., [["B", "L", "AA1", "K", "HH", "AW2", "S"], ...]
    Y_pred: list of ground truth phoneme sequences. e.g., [["B", "L", "AA1", "K", "HH", "AW2", "S"], ...]
    '''
    num_phonemes, num_errors = 0, 0
    for y_true, y_pred in zip(Y_true, Y_pred):
        num_phonemes += len(y_true)
        num_errors += levenshtein(y_true, y_pred)

    per = round(num_errors / num_phonemes, 2)
    return per, num_errors

def eval(model, iterator, device, dec_maxlen):
    model.eval()

    Y_true, Y_pred = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            x, x_seqlens, words, decoder_inputs, y, y_seqlens, prons = batch
            x, decoder_inputs = x.to(device), decoder_inputs.to(device)

            _, y_hat = model(x, x_seqlens, decoder_inputs, False, dec_maxlen) # <- teacher forcing is suppressed.

            y = y.to('cpu').numpy().tolist()
            y_hat = y_hat.to('cpu').numpy().tolist()
            for yy, yy_hat in zip(y, y_hat):
                y_true = convert_ids_to_phonemes(yy, idx2p)
                y_pred = convert_ids_to_phonemes(yy_hat, idx2p)
                Y_true.append(y_true)
                Y_pred.append(y_pred)

    # calc per.
    per, num_errors = calc_per(Y_true, Y_pred)
    print("per: %.2f" % per, "num errors: ", num_errors)

    with open("result", "w") as fout:
        for y_true, y_pred in zip(Y_true, Y_pred):
            fout.write(" ".join(y_true) + "\n")
            fout.write(" ".join(y_pred) + "\n\n")

    return per

#
# Train & Evaluate
#

train_dataset = G2pDataset(train_words, train_prons, g2idx, p2idx)
eval_dataset = G2pDataset(eval_words, eval_prons, g2idx, p2idx)

# (Pdb) eval_dataset[0]
# ([20, 7, 11, 21, 7, 4, 23, 21, 21, 7, 2], 11, 'r e i s e b u s s e', [2, 49, 9, 14, 5, 20, 47, 50, 5], [49, 9, 14, 5, 20, 47, 50, 5, 3], 9, 'ʁ aj z ə b ʊ s ə')

# x             = [20, 7, 11, 21, 7, 4, 23, 21, 21, 7, 2]
#                  r   e  i   s   e  b  u   s   s   e </s>
# x_seqlen      = 11
# word          = 'r e i s e b u s s e'
# decoder_input = [2, 49, 9, 14, 5, 20, 47, 50, 5]
# y             = [49, 9, 14, 5, 20, 47, 50, 5, 3]
# y_seqlen      = 9
# pron          = 'ʁ aj z ə b ʊ s ə'


train_iter = data.DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=pad)
eval_iter = data.DataLoader(eval_dataset, batch_size=hp.batch_size, shuffle=False, collate_fn=pad)

# breakpoint()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


encoder = Encoder(hp.emb_units, hp.hidden_units, g2idx)
decoder = Decoder(hp.emb_units, hp.hidden_units, p2idx)
model = Net(encoder, decoder)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr = hp.lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(1, hp.num_epochs+1):
    print(f"\nepoch: {epoch}")
    train(model, train_iter, optimizer, criterion, device)
    eval(model, eval_iter, device, hp.dec_maxlen)

torch.save(model.state_dict(), MODEL_PATH)
print (f"{MODEL_PATH} written.")

