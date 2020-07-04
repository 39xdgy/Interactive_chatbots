from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

example = "Cats Running was"

example = [stemmer.stem(token) for token in example.split(" ")]
print(" ".join(example))
