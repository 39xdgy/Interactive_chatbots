import spacy

nlp = spacy.load('en_core_web_sm')

example1 = nlp("Animals")
for token in example1:
    print(token.lemma_)

print()

example2 = nlp("I am god")
for token in example2:
    print(token.lemma_)
