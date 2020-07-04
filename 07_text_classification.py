from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

sentences = [
    "This is an awesome moive, I totally love it.",
    "What a waste of time and money",
    "This is my house",
    "I'm going to block you",
    "I love you",
    "Your product is by far the worst I've ever used"
]

for sentence in sentences:
    print(analyzer.polarity_scores(sentence)['compound'])
