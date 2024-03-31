#!/bin/env python3

from pathlib import Path
import yaml

import torch
from model import Encoder, Decoder, Net, load_vocab
from g2pdata import encode, convert_ids_to_phonemes

import numpy as np

#LEXICON_DE = Path('/home/guenter/projects/hal9000/ennis/chat/src/efficientspeech/lexicon/german_mfa.dict')

MODEL_PATH = Path('g2p_de.ckpt')
CONFIG_PATH = Path('../g2p_de/de_tiny.yaml')
#NPZ_PATH = Path('checkpoint_de.npz')

# DEBUG_LIMIT = 23
DEBUG_LIMIT = 0

hp = yaml.load( open(CONFIG_PATH, "r"), Loader=yaml.FullLoader)

g2idx, idx2g, p2idx, idx2p = load_vocab(hp)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(hp['model']['emb_units'], hp['model']['hidden_units'], g2idx)
decoder = Decoder(hp['model']['emb_units'], hp['model']['hidden_units'], p2idx)
model = Net(encoder, decoder)

print (f"loading model from {str(MODEL_PATH)} ...")

state_dict = torch.load(MODEL_PATH)

for k in state_dict:
    print (f"{k}: {state_dict[k].shape}")

# encoder.emb.weight: torch.Size([33, 64])
# encoder.rnn.weight_ih_l0: torch.Size([384, 64])
# encoder.rnn.weight_hh_l0: torch.Size([384, 128])
# encoder.rnn.bias_ih_l0: torch.Size([384])
# encoder.rnn.bias_hh_l0: torch.Size([384])
#
# decoder.emb.weight: torch.Size([56, 64])
# decoder.rnn.weight_ih_l0: torch.Size([384, 64])
# decoder.rnn.weight_hh_l0: torch.Size([384, 128])
# decoder.rnn.bias_ih_l0: torch.Size([384])
# decoder.rnn.bias_hh_l0: torch.Size([384])
# decoder.fc.weight: torch.Size([56, 128])
# decoder.fc.bias: torch.Size([56])

model.load_state_dict(state_dict)

model.to(device)

#(Pdb) print(batch[1])
#(tensor([[10,  3, 14,  ...,  0,  0,  0],
#        [ 5, 17, 16,  ...,  0,  0,  0],
#        [ 4,  3, 23,  ...,  0,  0,  0],
#        ...,
#        [21, 18, 11,  ...,  0,  0,  0],
#        [14, 17, 21,  ...,  0,  0,  0],
#        [13,  3, 21,  ...,  0,  0,  0]]), [8, 10, 9, 13, 12, 9, 11, 7, 7, 7, 10, 11, 5, 11, 9, 20, 18, 13, 10, 4, 11, 14, 10, 9, 12, 10, 10, 16, 10, 11, 10, 14, 10, 11, 16, 13, 7, 7, 12, 14, 18, 12, 12, 13, 10, 9, 9, 15, 10, 7, 8, 13, 10, 9, 14, 11, 12, 22, 17, 16, 10, 6, 12, 6, 11, 11, 20, 8, 18, 11, 12, 10, 10, 8, 14, 15, 13, 13, 13, 19, 8, 15, 12, 6, 10, 13, 10, 12, 17, 18, 12, 8, 11, 5, 12, 9, 11, 9, 19, 11, 6, 9, 13, 14, 11, 11, 7, 8, 6, 12, 12, 7, 11, 7, 8, 11, 18, 13, 8, 10, 11, 19, 14, 8, 12, 13, 12, 11], ['h a l t l o s', 'c o n t r a c t s', 'b a u r e i h e', 's t e r n f ö r m i g e', 'g e l d k a p i t a l', 't r a i n e r s', 'd r a m a s e r i e', 's t e r b e', 's i p i l ä', 'a n f a l l', 'w e s t l i c h e', 'a u f g r e i f e n', 'l y n x', 'v o l k s a l t a r', 'g a t t e r e r', 'g e s c h i c h t s s c h r e i b e r', 'g r u p p e n d y n a m i s c h e', 'a u f g e f o r s t e t', 'm e r g e s o r t', 't e d', 'z e u g n i s s e s', 'l u d w i g s b u r g e r', 'm a r i a z e l l', 'v e r l i n k t', 'b e d r o h u n g e n', 'k o n t r o l l e', 's ä g e b l a t t', 'd e m o g r a p h i s c h e n', 'h a i n b u c h e', 'v e r k l a p p e n', 'n i c h t s t u n', 's c h i e s s a n l a g e', 'k o r r e k t e r', 's i c h e r s t e n', 'n a h r u n g s p f l a n z e', 'f o r s t p o l i t i k', 'k o c h e n', 'l a r r e a', 'h o c h m e i s t e r', 'g l o r r e i c h s t e n', 'k r e i s v o r s i t z e n d e n', 'o r t h o p a e d i c', 'z u r ü c k f ü h r t', 'a r c h i v d i e n s t', 'k u c z y n s k i', 'e r s c h r a k', 'r o m a n d i e', 'r e k a p i t u l i e r e n', 'f e i e r o h m d', 't o m b a k', 'e r t a p p t', 'e r m ö g l i c h t e n', 'd o n a u r a u m', 'd e f o r m e d', 'f u n k e n s t r e c k e', 'z a i r i s c h e n', 's c h w a b b e l i g', 'w i n d g e s c h w i n d i g k e i t e n', 'h a u p t i n s t r u m e n t e', 'p r o f i f u s s b a l l e r', 'g e k u p p e l t', 'p e a c e', 'f u n d s c h i c h t', 'd i n g o', 'n e b e n f i g u r', 'a n g e k n i p s t', 'p e r s o n a l a u s s t a t t u n g', 'g e t r ü b t', 'o f f i z i e r s l a u f b a h n', 'e u r o l e a g u e', 'ü b e r s t i m m e n', 'w ä h r u n g e n', 'h y p e r ä m i e', 'c h a m p e x', 'f i r m e n a n g a b e n', 's c h w a r z e n e g g e r', 'g r o s s h a s l a c h', 'g i t a r r e n s o l o', 'c a m p i n g s t u h l', 'i n s t i t u t i o n a l i s m u s', 'd e f e k t e', 'c r i c k e t s p i e l e r', 'm i t t e l s t a d t', 'l e b e n', 'e r m o r d u n g', 's a n d s t r a n d e s', 'w a h l l i s t e', 'h e t e r o g e n e s', 'ö s t e r r e i c h i s c h e s', 's c h w e r p u n k t m ä s s i g', 'n a c h t a k t i v e', 'm a l t e s e', 'v o r g e f ü h r t', 't o b a', 't h e a t e r s a a l', 't r a v i a t a', 'a b s p r a c h e n', 'r e v i s i o n', 'h o c h s c h u l a s s i s t e n t', 'h a n d w a f f e n', 'd i a n a', 'q i a n l o n g', 'a u f z e i c h n u n g', 'b r i t a n n i s c h e n', 'z u n e h m e n d e', 'a b g e s e g n e t', 'y o u s e f', 'e u p h r a t', 'm o u n d', 'p r o j e k t p l a n', 'v e r s t ä r k u n g', 'c a r u s o', 'j o u r n a l i s m', 'p u h d y s', 'e r l a g e n', 'ü b e r n o m m e n', 'l a n d e s k r i m i n a l a m t', 't h e a t e r m a l e r', 'h e r z e n s', 'h a r r a c h o v', 'h e i l i g s t e n', 'f o r t b i l d u n g s s c h u l e', 's ü s s k a r t o f f e l', 'g r e m l i n', 'k u t s c h i e r e n', 's p i e l e v e r l a g', 'l o s g e w o r d e n', 'k a s s e r o l l e'], tensor([[ 2, 23, 39,  ...,  0,  0,  0],
#        [ 2, 22, 25,  ...,  0,  0,  0],
#        [ 2, 20, 44,  ...,  0,  0,  0],
#        ...,
#        [ 2, 37, 32,  ...,  0,  0,  0],
#        [ 2, 40,  7,  ...,  0,  0,  0],
#        [ 2, 22, 39,  ...,  0,  0,  0]]), tensor([[23, 39, 40,  ...,  0,  0,  0],
#        [22, 25, 29,  ...,  0,  0,  0],
#        [20, 44, 49,  ...,  0,  0,  0],
#        ...,
#        [37, 32,  6,  ...,  0,  0,  0],
#        [40,  7, 50,  ...,  0,  0,  0],
#        [22, 39, 50,  ...,  0,  0,  0]]), [8, 10, 6, 13, 12, 8, 11, 7, 7, 6, 9, 9, 6, 11, 7, 13, 15, 12, 10, 4, 9, 13, 9, 9, 10, 9, 9, 13, 8, 10, 9, 10, 8, 9, 13, 13, 6, 6, 9, 11, 16, 11, 10, 11, 9, 7, 8, 14, 7, 7, 7, 12, 8, 9, 13, 8, 9, 19, 16, 13, 9, 6, 9, 6, 11, 11, 17, 8, 14, 10, 10, 8, 8, 8, 14, 11, 11, 12, 11, 20, 8, 12, 10, 6, 9, 13, 8, 12, 12, 15, 11, 8, 10, 5, 9, 9, 10, 9, 15, 10, 6, 7, 9, 11, 10, 11, 6, 6, 4, 12, 11, 7, 11, 6, 8, 9, 18, 10, 8, 8, 10, 16, 12, 8, 9, 12, 12, 9], ['h a l t l oː s', 'kʰ ɔ n t ʁ ɛ ɪ t s', 'b aw ʁ aj ə', 'ʃ t ɛ ʁ n f œ ʁ m ɪ ɡ ə', 'ɟ ɛ l t kʰ a pʰ ɪ tʰ aː l', 't ʁ ɛ ɪ n ɐ s', 'd ʁ a m aː z eː ʁ ɪ ə', 'ʃ t ɛ ʁ b ə', 'z ɪ pʰ ɪ l eː', 'a n f a l', 'v ɛ s t l ɪ ç ə', 'aw f ɡ ʁ aj f ə n', 'l ʏ ŋ k s', 'f ɔ l k s a l tʰ aː ɐ', 'ɡ a tʰ ə ʁ ɐ', 'ɡ ə ʃ ɪ ç t s ʃ ʁ aj b ɐ', 'ɡ ʁ ʊ pʰ ə n d ʏ n aː m ɪ ʃ ə', 'aw f ɡ ə f ɔ ʁ s t ə t', 'm ɛ ʁ ɡ ə s ɔ ʁ t', 'tʰ ɛ t', 'ts ɔʏ k n ɪ s ə s', 'l uː t v ɪ ç s b ʊ ʁ ɡ ɐ', 'm a ʁ iː a ts ɛ l', 'f ɛ ʁ l ɪ ŋ k t', 'b ə d ʁ oː ʊ ŋ ə n', 'kʰ ɔ n t ʁ ɔ l ə', 'z eː ɡ ə b l a t', 'd ɛ m ɔ ɡ ʁ aː f ɪ ʃ ə n', 'h aj n b uː x ə', 'f ɛ ɐ k l a pʰ ə n', 'n ɪ ç t ʃ t uː n', 'ʃ iː s a n l aː ɡ ə', 'kʰ ɔ ʁ ɛ k tʰ ɐ', 'z ɪ ç ɐ s t ə n', 'n aː ʁ ʊ ŋ s pf l a n ts ə', 'f ɔ ʁ s t pʰ ɔ l ɪ tʰ iː k', 'kʰ ɔ x ə n', 'l aː ʁ eː a', 'h oː x m aj s t ɐ', 'ɡ l oː ʁ aj ç s t ə n', 'k ʁ aj s f oː ɐ z ɪ ts ə n d ə n', 'ɔ ʁ tʰ ɔ pʰ a ɛ d ɪ k', 'ts ʊ ʁ ʏ k f yː ɐ t', 'a ʁ ç ɪ v d iː n s t', 'kʰ ʊ ts yː n s c ɪ', 'ɛ ɐ ʃ ʁ a k', 'ʁ ɔ m a n d iː', 'ʁ ɛ kʰ a pʰ ɪ tʰ ʊ l iː ʁ ə n', 'f aj ɐ oː m t', 'tʰ ɔ m b a k', 'ɛ ɐ tʰ ɛ p t', 'ɛ ɐ m øː k l ɪ ç tʰ ə n', 'd oː n aw ʁ aw m', 'd ɛ f ɔ ʁ m ɛ t', 'f ʊ ŋ kʰ ə n ʃ t ʁ ɛ kʰ ə', 'ts aj ʁ ɪ ʃ ə n', 'ʃ v a b ə l ɪ ç', 'v ɪ n t ɡ ə ʃ v ɪ n d ɪ ç kʰ aj tʰ ə n', 'h aw p tʰ ɪ n s t ʁ ʊ m ɛ n tʰ ə', 'p ʁ oː f ɪ f uː s b a l ɐ', 'ɡ ə kʰ ʊ pʰ ə l t', 'pʰ iː n a s', 'f ʊ n t ʃ ɪ ç t', 'd ɪ ŋ ɡ ɔ', 'n eː b ə n f ɪ ɡ uː ɐ', 'a n ɡ ə k n ɪ p s t', 'pʰ ɛ ʁ z ɔ n aː l aw s ʃ t a tʰ ʊ ŋ', 'ɡ ə t ʁ yː p t', 'ɔ f ɪ ts iː ɐ s l aw f b aː n', 'ɔʏ ʁ ɔ l ɛ aː ɡ ʊ ə', 'yː b ɐ ʃ t ɪ m ə n', 'v eː ʁ ʊ ŋ ə n', 'h ʏ pʰ ɐ ɛ m iː', 'ʃ a m pʰ ɛ k s', 'f ɪ ʁ m ə n a n ɡ aː b ə n', 'ʃ v a ʁ ts ə n ɛ ɡ ɐ', 'ɡ ʁ oː s h a s l a x', 'ɟ ɪ tʰ a ʁ ə n z ɔ l ɔ', 'cʰ ɛ m pʰ ɪ ŋ ʃ t uː l', 'ɪ n s t ɪ tʰ uː t s ɪ ɔ n aː l ɪ s m ʊ s', 'd ɛ f ɛ k tʰ ə', 'k ʁ ɪ kʰ ə t ʃ p iː l ɐ', 'm ɪ tʰ ə l ʃ t a t', 'l eː b ə n', 'ɛ ɐ m ɔ ʁ d ʊ ŋ', 'z a n t ʃ t ʁ a n d ə s', 'v aː l ɪ s t ə', 'h ɛ tʰ ə ʁ ɔ ɟ ɛ n ə s', 'øː s t ə ʁ aj ç ɪ ʃ ə s', 'ʃ v eː ɐ pʰ ʊ ŋ k t m eː s ɪ ç', 'n a x tʰ a k tʰ iː v ə', 'm a l tʰ eː z ə', 'f oː ɐ ɡ ə f yː ɐ t', 'tʰ oː b a', 'tʰ ɛ aː tʰ ɐ z aː l', 't ʁ a v ɪ aː tʰ a', 'a p ʃ p ʁ aː x ə n', 'ʁ ɛ v ɪ z j oː n', 'h oː x ʃ uː l a s ɪ s t ɛ n t', 'h a n t v a f ə n', 'd ɪ aː n a', 'cʰ a n l ɔ ŋ', 'aw f ts aj ç n ʊ ŋ', 'b ʁ ɪ tʰ a n ɪ ʃ ə n', 'ts uː n eː m ɛ n d ə', 'a p ɡ ə z eː k n ə t', 'j uː z eː f', 'ɔʏ f ʁ aː t', 'm aw n', 'p ʁ ɔ j ɛ k t p l aː n', 'f ɛ ɐ ʃ t ɛ ʁ kʰ ʊ ŋ', 'kʰ a ʁ uː z oː', 'j oː ʊ ʁ n a l ɪ s m', 'pʰ uː d ɪ s', 'ɛ ɐ l aː ɡ ə n', 'yː b ɐ n ɔ m ə n', 'l a n d ə s k ʁ ɪ m ɪ n aː l a m t', 'tʰ ɛ aː tʰ ɐ m aː l ɐ', 'h ɛ ʁ ts ə n s', 'h a ʁ a x oː f', 'h aj l ɪ ç s t ə n', 'f ɔ ʁ t b ɪ l d ʊ ŋ s ʃ uː l ə', 'z yː s k a ʁ tʰ ɔ f ə l', 'ɡ ʁ ɛ m l iː n', 'kʰ ʊ tʃ ʃ iː ʁ ə n', 'ʃ p iː l ə f ɛ ɐ l aː k', 'l oː s ɡ ə v ɔ ɐ d ə n', 'kʰ a s ə ʁ ɔ l ə'])

model.eval()

def g2p_de(model, word, g2idx, idx2p):

    x = encode (word, "x", g2idx)
    #while len(x)<hp.enc_maxlen:
    #    x.append(0)
    print (f"{word} -> {x}")

    x_seqlens = torch.tensor([len(x)])
    x = torch.tensor([x])

    print (f"x_seqlens={x_seqlens}, x={x}")

    decoder_inputs = torch.tensor([encode("", "y", p2idx)])

    print (f"decoder_inputs={decoder_inputs}")

    _, y_hat = model(x, x_seqlens, decoder_inputs, False, hp.dec_maxlen) # <- teacher forcing is suppressed.

    print (f"y_hat={y_hat}")

    y_hat = y_hat.to('cpu').numpy().tolist()

    y_pred = convert_ids_to_phonemes(y_hat[0], idx2p)

    print (f"y_pred={y_pred}")

g2p_de(model, "E i s e n b a h n", g2idx, idx2p)
g2p_de(model, "B i m b o", g2idx, idx2p)
g2p_de(model, "Z e i t u n g", g2idx, idx2p)
g2p_de(model, "E i g e n t u m", g2idx, idx2p)



