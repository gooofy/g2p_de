# g2p_de: A Simple Python Module for German Grapheme To Phoneme Conversion

This module is designed to convert German graphemes (spelling) to phonemes (pronunciation).
It is a pretty crude adaptation of g2pE by Kyubyong Park & Jongseok Kim to German. See

https://github.com/Kyubyong/g2p

for further details about the algorithms used und scientific background information.

Note: This implementation does not deal with homographs yet.

## Algorithm

1. Spell out arabic numbers
3. Look up the word in [Montreal Forced Aligner's German Dictionary](https://mfa-models.readthedocs.io/en/latest/dictionary/German/German%20MFA%20dictionary%20v3_0_0.html#German%20MFA%20dictionary%20v3_0_0)
4. For OOVs, predict their pronunciations using a GRU based neural net model.

## Environment

* python 3.x

## Dependencies

* numpy >= 1.13.1
* nltk >= 3.2.4
* num2words >= 0.5.13

## Installation

    python setup.py install

## Usage

    from g2p_de import G2p

    if __name__ == '__main__':
        texts = ["Ich habe 250 Euro in meiner Tasche.", # number -> spell-out
                 "Verschiedene Haustiere, z.B. Hunde und Katzen", # z.B. -> zum Beispiel
                 "KI ist ein Teilgebiet der Informatik, das sich mit der Automatisierung intelligenten Verhaltens und dem maschinellen Lernen befasst.",
                 "Dazu gehören nichtsteroidale Antirheumatika (z. B. Acetylsalicylsäure oder Ibuprofen), Lithium, Digoxin, Dofetilid oder Fluconazol"]
        g2p = G2p()
        for text in texts:
            out = g2p(text)
            print(out)
        from g2p_de import G2p

result:

    ['ɪ', 'ç', ' ', 'h', 'aː', 'b', 'ə', ' ', 'ts', 'v', 'aj', 'h', 'ʊ', 'n', 'd', 'ɐ', 't', 'f', 'ʏ', 'n', 'f', 'ts', 'ɪ', 'ç', ' ', 'ɔʏ', 'ʁ', 'ɔ', ' ', 'ɪ', 'n', ' ', 'm', 'aj', 'n', 'ɐ', ' ', 'tʰ', 'a', 'ʃ', 'ə', ' ', '.']
    ['f', 'ɛ', 'ɐ', 'ʃ', 'iː', 'd', 'ə', 'n', 'ə', ' ', 'h', 'aw', 's', 't', 'iː', 'ʁ', 'ə', ' ', ',', ' ', 'ts', 'ʊ', 'm', ' ', 'b', 'aj', 'ʃ', 'p', 'iː', 'l', ' ', 'h', 'ʊ', 'n', 'd', 'ə', ' ', 'ʊ', 'n', 't', ' ', 'kʰ', 'a', 'ts', 'ə', 'n']
    ['cʰ', 'ɪ', ' ', 'ɪ', 's', 't', ' ', 'ə', 'n', ' ', 'tʰ', 'aj', 'l', 'ɡ', 'ə', 'b', 'iː', 't', ' ', 'd', 'eː', 'ʁ', ' ', 'ɪ', 'n', 'f', 'ɔ', 'ɐ', 'm', 'aː', 'tʰ', 'ɪ', 'k', ' ', ',', ' ', 'd', 'aː', 's', ' ', 'z', 'ɪ', 'ç', ' ', 'm', 'ɪ', 't', ' ', 'd', 'eː', 'ʁ', ' ', 'aw', 'tʰ', 'ɔ', 'm', 'aː', 'tʰ', 'ɪ', 'z', 'iː', 'ʁ', 'ʊ', 'ŋ', ' ', 'ɪ', 'n', 'tʰ', 'ɛ', 'l', 'ɪ', 'ɟ', 'ɛ', 'n', 'tʰ', 'ə', 'n', ' ', 'f', 'ɛ', 'ɐ', 'h', 'a', 'l', 'tʰ', 'ə', 'n', 's', ' ', 'ʊ', 'n', 't', ' ', 'd', 'ɛ', 'm', ' ', 'm', 'a', 'ʃ', 'ɪ', 'n', 'ɛ', 'l', 'ə', 'n', ' ', 'l', 'ɛ', 'ɐ', 'n', 'ə', 'n', ' ', 'b', 'ə', 'f', 'a', 's', 't', ' ', '.']
    ['d', 'aː', 'ts', 'uː', ' ', 'ɡ', 'ə', 'h', 'øː', 'ʁ', 'ə', 'n', ' ', 'n', 'ɪ', 'ç', 't', 'ʃ', 't', 'ɛ', 'ʁ', 'oː', 'ɪ', 'd', 'aː', 'l', 'ə', ' ', 'a', 'n', 'tʰ', 'ɪ', 'ʁ', 'ɔʏ', 'm', 'aː', 'tʰ', 'ɪ', 'kʰ', 'a', ' ', '(', ' ', 'ts', 'ɛ', 't', ' ', '.', ' ', 'b', 'eː', ' ', '.', ' ', 'a', 'ts', 'ɛ', 'tʰ', 'ʏ', 'l', 'a', 's', 'ɪ', 'l', 'ʏ', 'z', 'ɛ', 'k', 's', 'ə', ' ', 'oː', 'd', 'ɐ', ' ', 'iː', 'b', 'ʊ', 'p', 'ʁ', 'ɔ', 'f', 'eː', 'n', ' ', ')', ' ', ',', ' ', 'l', 'iː', 'tʰ', 'ɪ', 'ʊ', 'm', ' ', ',', ' ', 'd', 'ɪ', 'ɡ', 'ɔ', 'k', 's', 'iː', 'n', ' ', ',', ' ', 'd', 'ɔ', 'f', 'ɛ', 'tʰ', 'iː', 'l', 't', ' ', 'oː', 'd', 'ɐ', ' ', 'f', 'l', 'uː', 'kʰ', 'ɔ', 'n', 'a', 'ts', 'oː', 's']


## Training

    cd training
    ./train.py

| hidden | embedding | num_errors | run                | train % |
|--------|-----------|------------|--------------------|---------|
| 256    | 64        | 15050      | Mar31_13-48-03_hal | 80      |
| 128    | 128       | 19300      | Mar31_14-34-31_hal | 80      |
| 128    | 64        | 19512      | Mar31_15-27-23_hal | 80      |
| 128    | 64        |            | Mar31_16-17-16_hal | 90      |



