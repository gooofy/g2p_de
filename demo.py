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

