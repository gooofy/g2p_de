import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#class Hparams:
#    batch_size = 2048
#    enc_maxlen = 20
#    dec_maxlen = 20
#    # num_epochs = 42
#    num_epochs = 2
#    hidden_units = 128
#    emb_units = 64
#    graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyzüöäß")
#    phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['ts', 'ə', 'iː', 'oː', 'pf', 'aj', 'd', 'tʃ', 'm', 'œ', 'z', 'ɛ', 'ɲ', 't', 'ɟ', 'n̩', 'b', 'ɪ', 'kʰ', 'h', 'eː', 'ɔ', 'f', 'v', 'l̩', 'n', 'x', 'yː', 'p', 'c', 'aː', 'ç', 'uː', 'ʃ', 'øː', 'a', 'l', 'j', 'ɔʏ', 'cʰ', 'aw', 'ŋ', 'ɐ', 'ʊ', 'pʰ', 'ʁ', 's', 'ʏ', 'ɡ', 'tʰ', 'k', 'm̩']
#    lr = 0.001
#    logdir = "log/01"

def load_vocab(hp):
    g2idx = {g: idx for idx, g in enumerate(hp['model']['graphemes'])}
    idx2g = {idx: g for idx, g in enumerate(hp['model']['graphemes'])}

    p2idx = {p: idx for idx, p in enumerate(hp['model']['phonemes'])}
    idx2p = {idx: p for idx, p in enumerate(hp['model']['phonemes'])}

    return g2idx, idx2g, p2idx, idx2p # note that g and p mean grapheme and phoneme, respectively.

#
# Model
#

class Encoder(nn.Module):

    def __init__(self, emb_units, hidden_units, g2idx):
        super().__init__()
        self.emb_units = emb_units
        self.hidden_units = hidden_units
        self.emb = nn.Embedding(len(g2idx), emb_units)
        self.rnn = nn.GRU(emb_units, hidden_units, batch_first=True)

    def forward(self, x, seqlens):
        x = self.emb(x)

        # packing -> rnn -> unpacking -> position recovery: note that enforce_sorted is set to False.
        packed_input = pack_padded_sequence(x, seqlens, batch_first=True, enforce_sorted=False)
        outputs, last_hidden = self.rnn(packed_input)
#         outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=x.size()[1])

        # last hidden
        last_hidden = last_hidden.permute(1, 2, 0)
        last_hidden = last_hidden.view(last_hidden.size()[0], -1)

        return last_hidden

class Decoder(nn.Module):

    def __init__(self, emb_units, hidden_units, p2idx):
        super().__init__()

        self.emb_units = emb_units
        self.hidden_units = hidden_units
        self.emb = nn.Embedding(len(p2idx), emb_units)
        self.rnn = nn.GRU(emb_units, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, len(p2idx))

    def forward(self, decoder_inputs, h0):
        decoder_inputs = self.emb(decoder_inputs)

        outputs, last_hidden = self.rnn(decoder_inputs, h0)
        logits = self.fc(outputs) # (N, T, V)
        y_hat = logits.argmax(-1)

        return logits, y_hat, last_hidden

class Net(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, seqlens, decoder_inputs, teacher_forcing=True, dec_maxlen=None):
        '''
        At training, decoder inputs (ground truth) and teacher forcing is applied.
        At evaluation, decoder inputs are ignored, and the decoding keeps for `dec_maxlen` steps.
        '''
        last_hidden = self.encoder(x, seqlens)
        h0 = last_hidden.unsqueeze(0)

        if teacher_forcing: # training
            logits, y_hat, h0 = self.decoder(decoder_inputs, h0)
        else: # evaluation
            decoder_inputs = decoder_inputs[:, :1] # "<s>"
            logits, y_hat = [], []
            for t in range(dec_maxlen):
                _logits, _y_hat, h0 =self.decoder(decoder_inputs, h0) # _logits: (N, 1, V), _y_hat: (N, 1), h0: (1, N, N)
                logits.append(_logits)
                y_hat.append(_y_hat)
                decoder_inputs = _y_hat

            logits = torch.cat(logits, 1)
            y_hat = torch.cat(y_hat, 1)

        return logits, y_hat

