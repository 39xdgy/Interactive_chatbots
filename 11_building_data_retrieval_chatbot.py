import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("test_QA.csv")
df.dropna(inplace = True)
#print(df)

vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((df.Questions, df.Answers)))

Question_vectors = vectorizer.transform(df.Questions)

print("You can start chatting with me now")
while True:
    input_question = input()
    if(input_question == 'Q'):
        break
    input_question_vector = vectorizer.transform([input_question])

    similarities = cosine_similarity(input_question_vector, Question_vectors)

    closest = np.argmax(similarities, axis = 1)

    print("BOT: " + df.Answers.iloc[closest].values[0])
