# import nltk
# import pickle
# import numpy as np
# from nltk.stem import WordNetLemmatizer
# from keras.models import load_model
# from app.config import MODEL_PATH, TRAINING_DATA_PATH

# lemmatizer = WordNetLemmatizer()
# model = load_model(MODEL_PATH)
# data = pickle.load(open(TRAINING_DATA_PATH, "rb"))
# words = data['words']
# classes = data['classes']

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

# def bow(sentence, words):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#     # FIX: Ensure it's a 2D NumPy array
#     return np.array([bag])

# def predict_class(sentence):
#     input_data = bow(sentence, words)
#     print(f"Recognized Bag: {input_data}")
#     res = model.predict(input_data, verbose=0)[0]
#     print(f"Confidences: {res}") # verbose=0 keeps logs clean
#     results = [[i, r] for i, r in enumerate(res) if r > 0.25]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]






import nltk
import pickle
import numpy as np
import os
from nltk.stem import PorterStemmer # Switch back to Stemmer
from keras.models import load_model
from app.config import MODEL_PATH, TRAINING_DATA_PATH

stemmer = PorterStemmer()

# Load model and data
model = load_model(MODEL_PATH)
data = pickle.load(open(TRAINING_DATA_PATH, "rb"))
words = data['words']
classes = data['classes']

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    # MUST match the training script: stem and lowercase
    sentence_words = [stemmer.stem(w.lower()) for w in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    # The 2D array fix for TensorFlow
    return np.array([bag])

# Use this in your chatbot.py
def predict_class(sentence):
    input_data = bow(sentence, words)
    res = model.predict(input_data, verbose=0)[0]
    
    # Debug print to your terminal so you can see the confidence!
    # print(f"Intent Confidences: {res}") 
    
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]