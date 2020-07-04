import spacy

nlp = spacy.load('en_core_web_sm')

example1 = nlp("This is an example of text tokenization")

for token in example1:
    print(token.text)

print()

example2 = nlp("The quick brown fox jumps over the lazy dog")

for token in example2:
    print(token.text)

print()

example3 = nlp("We're the champions")

for token in example3:
    print(token.text)

print()

example4 = nlp("Will we have dinner today?")

for token in example4:
    print(token.text)
