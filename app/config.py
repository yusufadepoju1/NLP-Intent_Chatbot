import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.h5")
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "models", "training_data")
ERROR_THRESHOLD = 0.25