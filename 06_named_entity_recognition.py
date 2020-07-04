import spacy

nlp = spacy.load('en_core_web_sm')

example = "Google, a company founded by Larry Page and Sergey Brin in the United States of America "\
+ "has one of the world's most advanced search engines. "

doc = nlp(example)

for ent in doc.ents:
    print(ent.text, ent.label_)
