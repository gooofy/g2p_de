from torch.utils import data

#
# Data Loader
#

def encode(inp, kind, d:dict):
    '''convert string into ids
    kind: "x" or "y"
    d: g2idx for 'x', p2idx for 'y'
    '''
    if kind=="x": tokens = inp.lower().split() + ["</s>"]
    else: tokens = ["<s>"] + inp.split() + ["</s>"]

    x = [d.get(t, d["<unk>"]) for t in tokens]
    return x

class G2pDataset(data.Dataset):

    def __init__(self, words, prons, g2idx, p2idx):
        """
        words: list of words. e.g., ["w o r d", ]
        prons: list of prons. e.g., ['W ER1 D',]
        """
        self.words = words
        self.prons = prons
        self._g2idx = g2idx
        self._p2idx = p2idx

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word, pron = self.words[idx], self.prons[idx]
        x = encode(word, "x", self._g2idx)
        y = encode(pron, "y", self._p2idx)
        decoder_input, y = y[:-1], y[1:]

        x_seqlen, y_seqlen = len(x), len(y)

        return x, x_seqlen, word, decoder_input, y, y_seqlen, pron

def convert_ids_to_phonemes(ids, idx2p):
    phonemes = []
    for idx in ids:
        if idx == 3: # 3: </s>
            break
        p = idx2p[idx]
        phonemes.append(p)
    return phonemes

