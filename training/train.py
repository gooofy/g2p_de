#!/bin/env python3

# adapted to german mfa from: https://github.com/Kyubyong/nlp_made_easy/blob/master/PyTorch%20seq2seq%20template%20based%20on%20the%20g2p%20task.ipynb

import os
import sys

import numpy as np
from tqdm import tqdm
from distance import levenshtein
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from model import Encoder, Decoder, Net, load_vocab
from g2pdata import G2pDataset, convert_ids_to_phonemes

CONTINUE_TRAINING = False

LEXICON_DE = Path('../g2p_de/german_mfa.dict')

CONFIG_PATH = Path('../g2p_de/de_tiny.yaml')
MODEL_PATH = Path('g2p_de.ckpt')
NPZ_PATH = Path('checkpoint_de.npz')

# DEBUG_LIMIT = 23
DEBUG_LIMIT = 0

def load_lex(lexicon_path:Path, hp):

    lexicon = {} # word -> [ phonemes ]

    with open (lexicon_path, 'r') as lexf:
        for line in lexf:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            graph = parts[0]

            if graph in lexicon:
                continue

            phonemes = parts[1].split(' ')
            valid = True
            for c in graph:
                if not c in hp['model']['graphemes']:
                    valid = False
                    break

            if not valid:
                continue

            # print (f"{graph} : {phonemes}")

            lexicon[graph] = phonemes

            if DEBUG_LIMIT and len(lexicon) >= DEBUG_LIMIT:
                break

    return lexicon

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
    num_train, num_eval = int(len(words)*.9), int(len(words)*.1)
    train_words, eval_words = words[:num_train], \
                              words[-num_eval:]
    train_prons, eval_prons = prons[:num_train], \
                              prons[-num_eval:]
    return train_words, eval_words, train_prons, eval_prons

def drop_lengthy_samples(words, prons, enc_maxlen, dec_maxlen):
    """We only include such samples less than maxlen."""
    _words, _prons = [], []
    for w, p in zip(words, prons):
        if len(w.split()) + 1 > enc_maxlen: continue
        if len(p.split()) + 1 > dec_maxlen: continue # 1: <EOS>
        _words.append(w)
        _prons.append(p)
    return _words, _prons

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

    return loss.item()

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
    # print("    percentage: %.2f" % per, "num errors: ", num_errors)

    with open("result.txt", "w") as fout:
        for y_true, y_pred in zip(Y_true, Y_pred):
            fout.write(" ".join(y_true) + "\n")
            fout.write(" ".join(y_pred) + "\n\n")
    #print ("    result.txt written.")

    return per, num_errors

def save_model (state_dict):

    torch.save(state_dict, MODEL_PATH)
    # print (f"{MODEL_PATH} written.")

    np.savez (NPZ_PATH, enc_emb  = state_dict['encoder.emb.weight'].cpu(),
                        enc_w_ih = state_dict['encoder.rnn.weight_ih_l0'].cpu(),
                        enc_w_hh = state_dict['encoder.rnn.weight_hh_l0'].cpu(),
                        enc_b_ih = state_dict['encoder.rnn.bias_ih_l0'].cpu(),
                        enc_b_hh = state_dict['encoder.rnn.bias_hh_l0'].cpu(),
                        dec_emb  = state_dict['decoder.emb.weight'].cpu(),
                        dec_w_ih = state_dict['decoder.rnn.weight_ih_l0'].cpu(),
                        dec_w_hh = state_dict['decoder.rnn.weight_hh_l0'].cpu(),
                        dec_b_ih = state_dict['decoder.rnn.bias_ih_l0'].cpu(),
                        dec_b_hh = state_dict['decoder.rnn.bias_hh_l0'].cpu(),
                        fc_w     = state_dict['decoder.fc.weight'].cpu(),
                        fc_b     = state_dict['decoder.fc.bias'].cpu())

    #print (f"{NPZ_PATH} written.")


hp = yaml.load( open(CONFIG_PATH, "r"), Loader=yaml.FullLoader)

g2idx, idx2g, p2idx, idx2p = load_vocab(hp)

# print (f"g2idx={g2idx}")
# print (f"p2idx={p2idx}")


lexicon = load_lex(LEXICON_DE, hp)

train_words, eval_words, train_prons, eval_prons = prepare_data(lexicon)

print (f"# entries: train: {len(train_words)}, eval: {len(eval_words)}")

train_words, train_prons = drop_lengthy_samples(train_words, train_prons, hp['model']['enc_maxlen'], hp['model']['dec_maxlen'])
# We do NOT apply this constraint to eval dataset.

print (f"# entries after dropping lenghty samples: train: {len(train_words)}, eval: {len(eval_words)}")


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

train_iter = data.DataLoader(train_dataset, batch_size=hp['training']['batch_size'], shuffle=True, collate_fn=pad)
eval_iter = data.DataLoader(eval_dataset, batch_size=hp['training']['batch_size'], shuffle=False, collate_fn=pad)

# breakpoint()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(hp['model']['emb_units'], hp['model']['hidden_units'], g2idx)
decoder = Decoder(hp['model']['emb_units'], hp['model']['hidden_units'], p2idx)
model = Net(encoder, decoder)

if CONTINUE_TRAINING and os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    print (f"model state {MODEL_PATH} loaded.")

model.to(device)

optimizer = optim.Adam(model.parameters(), lr = hp['training']['lr'])
criterion = nn.CrossEntropyLoss(ignore_index=0)

writer = SummaryWriter()

t = tqdm(range(1, hp['training']['num_epochs']+1))

best_errs = sys.maxsize
best_epoch = 0

for epoch in t:
    # print(f"\nepoch: {epoch}/{hp['training']['num_epochs']}")
    loss = train(model, train_iter, optimizer, criterion, device)
    writer.add_scalar('loss', loss, epoch)
    #print(f"    loss={loss}")

    _, num_errors = eval(model, eval_iter, device, hp['model']['dec_maxlen'])

    t.set_description(f"loss={loss:.3f}, errs={num_errors}[{best_errs}@{best_epoch}], lr={optimizer.param_groups[0]['lr']}")

    writer.add_scalar('errs', num_errors, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    if num_errors < best_errs:
        best_errs = num_errors
        best_epoch = epoch
        save_model(model.state_dict())


