import spacy

nlp = spacy.load('en_core_web_lg')

example1 = "man woman king queen"
tokens = nlp(example1)
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

example2 = "walking walked swimming swam"
tokens = nlp(example2)
for token1 in tokens:
    for token2 in tokens:
        if (token1.text == token2.text):
            continue
        print(token1.text, token2.text, token1.similarity(token2))

example3 = "spain russia madrid moscow"
tokens = nlp(example3)
for token1 in tokens:
    for token2 in tokens:
        if (token1.text == token2.text):
            continue
        print(token1.text, token2.text, token1.similarity(token2))
