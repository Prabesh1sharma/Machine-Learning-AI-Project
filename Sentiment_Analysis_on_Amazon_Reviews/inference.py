import numpy as np 
import neattext.functions as nfx
import nltk
from nltk.stem.porter import PorterStemmer
import re
import pickle
# nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [nfx.remove_stopwords(word) for word in review]
    review = [stemmer.stem(word) for word in review]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

# def single_prediction(predictor, scaler, cv, text_input):
#     corpus = []
#     stemmer = PorterStemmer()
#     review = re.sub("[^a-zA-Z]", " ", text_input)
#     review = review.lower().split()
#     review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
#     review = " ".join(review)
#     corpus.append(review)
#     X_prediction = cv.transform(corpus).toarray()
#     X_prediction_scl = scaler.transform(X_prediction)
#     y_predictions = predictor.predict_proba(X_prediction_scl)
#     y_predictions = y_predictions.argmax(axis=1)[0]

#     return "Positive" if y_predictions == 1 else "Negative"

def predict():
    predictor = pickle.load(open(r"model_rf.pkl", "rb"))
    scaler = pickle.load(open(r"scaler.pkl", "rb"))
    cv = pickle.load(open(r"countVectorizer.pkl", "rb"))
    text_input = input("Enter the review to check : ")
    predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
    if predicted_sentiment=='Positive':
        print("\nGiven Review is positive")
    else:
        print("\nGiven Review is Negative")

predict()