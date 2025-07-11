# src/train.py
import spacy
import random
import os
from spacy.training import Example
from . import config
from .data_handler import create_training_data

def train_ner_model():
    TRAIN_DATA = create_training_data()
    if not TRAIN_DATA:
        print("No training data. Exiting.")
        return

    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    
    for label in config.LABELS:
        ner.add_label(label)

    optimizer = nlp.begin_training()
    print("Starting model training...")

    for itn in range(config.N_ITERATIONS):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for item in TRAIN_DATA:
            text = item['text']
            annotations = {'entities': item['entities']}
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.35, sgd=optimizer, losses=losses)
        print(f"Iteration {itn + 1}/{config.N_ITERATIONS}, Losses: {losses}")

    if not os.path.exists(os.path.dirname(config.MODEL_PATH)):
        os.makedirs(os.path.dirname(config.MODEL_PATH))
    nlp.to_disk(config.MODEL_PATH)
    print(f"\nTraining complete. Model saved to '{config.MODEL_PATH}'")

if __name__ == "__main__":
    train_ner_model()
