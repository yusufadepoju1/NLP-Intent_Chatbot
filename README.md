# NLP Intent-Based Chatbot

A specialized conversational AI chatbot built with **Python**, **Flask**, **TensorFlow/Keras**, and **NLTK**. This project leverages Natural Language Processing (NLP) techniques, specifically stemming and bag-of-words (BoW) models, to classify user intents and provide appropriate responses. 

The chatbot is currently configured to discuss specific topics like a portfolio, AI projects, and movie recommendations, complete with a fallback mechanism when the model confidence is low.

---

##  Features

- **Deep Learning Model**: Utilizes a custom TensorFlow/Keras Sequential neural network for intent classification.
- **Natural Language Processing**: Employs NLTK for tokenization and Porter Stemming (`PorterStemmer`) to process and normalize user inputs.
- **RESTful API via Flask**: A lightweight web server providing a frontend UI (`/` endpoint) and an API endpoint (`/get`) for handling asynchronous chat messages.
- **Dynamic Responses**: Selects randomized but appropriate responses defined in the `intents.json` file based on predicted intent tags.
- **Confidence Thresholding**: Implements an error threshold to ensure the chatbot only responds when it is reasonably certain, alongside a structured fallback response.

---

##  Project Structure

```text
NLP-Intent_Chatbot/
│
├── app/
│   ├── config.py         # Configuration variables (paths to models, data, etc.)
│   ├── main.py           # Entry point for the Flask web server
│   ├── chatbot.py        # Core logic tying intent prediction to response selection
│   ├── inference.py      # Model inference logic, text preprocessing, and intent prediction
│   └── preprocessing.py  # Data preprocessing utilities
│
├── data/
│   └── intents.json      # The training data representing intents, queries, and responses
│
├── models/               # Directory where trained neural network models are saved
│
├── templates/
│   └── index.html        # Main HTML frontend template for the chat interface
│
├── train.py              # Script to train the Keras neural network on intents.json
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation (this file)
```

---

## Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd "NLP-Intent_Chatbot - Copy"
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Make sure NLTK requirements are downloaded. You may need to run `nltk.download('punkt')` in a Python shell if prompted.*

---

## Training the Model

Before starting the web application, you must train the model based on the intents provided in `data/intents.json`. 

Run the following command:
```bash
python train.py
```
This script will:
- Parse `intents.json` and extract vocabulary and tags.
- Apply Porter Stemming to the text data.
- Train a Dense Neural Network with dropout layers for intent classification.
- Save the trained weights to the `models/` directory.
- Save the vocabulary and classes into a pickled metadata file.

---

## Running the Application

Start the Flask server:
```bash
python -m app.main
```
or 
```bash
python app/main.py
```

The server will start on `http://127.0.0.1:5000/`. You can open this URL in your web browser to interact with the chatbot frontend.

---

## How it Works

1. **User Input Phase**: The user types a message in the browser UI, sending an asynchronous POST request to `/get`.
2. **Preprocessing**: The `inference.py` script receives the text, tokenizes it into words, drops punctuation, and stems the text.
3. **Bag of Words Generation**: The processed input is converted into a structured array consisting of `0`s and `1`s indicating the presence of words known to the vocabulary.
4. **Prediction**: The TensorFlow neural network predicts the probability list corresponding to different intent classes.
5. **Response Matching**: The `chatbot.py` file evaluates the predicted intent. If confidence > `0.5`, it retrieves a random matching response from `intents.json`. Otherwise, an intelligent fallback response is used.

---

## Customizing the Bot

To change what the chatbot talks about:
1. Open up `data/intents.json`.
2. Add new `tag`, `patterns` (example phrases users might say), and `responses`.
3. Save the JSON file.
4. Re-run `python train.py` to bake your new intents into the neural network.
5. Restart your Flask application.
