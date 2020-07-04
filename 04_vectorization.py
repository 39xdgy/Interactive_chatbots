from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(binary = True, token_pattern = r'\b[^\d\W]+\b')

corpus = ["The dog is on the tabel", "the cats now are on the tabel"]
vectorizer.fit(corpus)
print(vectorizer.transform(["The dog is on the tabel"]).toarray())

vocab = vectorizer.vocabulary_

for key in sorted(vocab.keys()):
    print("{}: {}".format(key, vocab[key]))


corpus2 = ["I am jack", "You are john", "I am john"]
vectorizer.fit(corpus2)

print(vectorizer.transform(corpus2).toarray())

vocab = vectorizer.vocabulary_

for key in sorted(vocab.keys()):
    print("{}: {}".format(key, vocab[key]))
