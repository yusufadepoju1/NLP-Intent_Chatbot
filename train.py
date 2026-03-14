# import os
# import json
# import pickle
# import numpy as np
# import nltk
# import random
# from nltk.stem import WordNetLemmatizer # Use Lemmatizer to match inference.py
# import tensorflow as tf
# from app.config import MODEL_PATH, TRAINING_DATA_PATH, INTENTS_PATH

# lemmatizer = WordNetLemmatizer()

# # Ensure model directory exists
# os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# # Load intents
# with open(INTENTS_PATH) as f:
#     intents = json.load(f)

# words = []
# classes = []
# documents = []
# ignore_letters = ['!', '?', ',', '.']

# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         # Tokenize
#         w = nltk.word_tokenize(pattern)
#         words.extend(w)
#         documents.append((w, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatize and sort
# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
# words = sorted(list(set(words)))
# classes = sorted(list(set(classes)))

# training = []
# output_empty = [0] * len(classes)

# for doc in documents:
#     bag = []
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)

#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
#     training.append([bag, output_row])

# # Shuffle and convert to numpy
# random.shuffle(training)
# training = np.array(training, dtype=object)

# train_x = np.array(list(training[:, 0]))
# train_y = np.array(list(training[:, 1]))

# # Improved Model
# # model = tf.keras.Sequential([
# #     tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
# #     tf.keras.layers.Dropout(0.5),
# #     tf.keras.layers.Dense(64, activation='relu'),
# #     tf.keras.layers.Dropout(0.5),
# #     tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
# # ])

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(10,input_shape=(len(train_x[0]),)))
# model.add(tf.keras.layers.Dense(10))
# model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))
# model.compile(tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=1)

# # Save everything using paths from config
# model.save(MODEL_PATH)
# pickle.dump({'words': words, 'classes': classes}, open(TRAINING_DATA_PATH, "wb"))

# print("Successfully trained and saved model structure.")









import os
import json
import pickle
import numpy as np
import nltk
import random
from nltk.stem import PorterStemmer
import tensorflow as tf
from app.config import MODEL_PATH, TRAINING_DATA_PATH, INTENTS_PATH

stemmer = PorterStemmer()

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load intents
with open(INTENTS_PATH) as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Stemming and sorting
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [stemmer.stem(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Optimized Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save(MODEL_PATH)
pickle.dump({'words': words, 'classes': classes}, open(TRAINING_DATA_PATH, "wb"))

print("\n[SUCCESS] Model trained with Stemming. Vocabulary and Weights saved.")