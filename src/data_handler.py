# src/data_handler.py
import pandas as pd
from docx import Document
from PIL import Image
import pytesseract
import os
import re
from . import config

def get_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def get_text_from_png(file_path):
    return pytesseract.image_to_string(Image.open(file_path))

def extract_text(file_path):
    _, extension = os.path.splitext(file_path)
    if extension.lower() == ".docx":
        return get_text_from_docx(file_path)
    elif extension.lower() == ".png":
        return get_text_from_png(file_path)
    return ""

import spacy
from spacy.matcher import Matcher

def create_training_data():
    """
    Creates annotated training data in spaCy format.
    THIS IS A MORE ADVANCED VERSION using spaCy's Matcher for robust entity finding.
    """
    try:
        train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Training data not found at {config.TRAIN_DATA_PATH}")
        return []

    train_df.columns = [col.strip().lower().replace(" ", "_") for col in train_df.columns]
    
    # Load a blank nlp model for tokenization
    nlp = spacy.blank("en")
    training_data = []
    print("Preparing training data with advanced matcher...")

    for _, row in train_df.iterrows():
        base_name = str(row['file_name'])
        possible_exts = ['.docx', '.png']
        file_path = None
        for ext in possible_exts:
            candidate = os.path.join(config.TRAIN_FILES_PATH, base_name + ext)
            if os.path.exists(candidate):
                file_path = candidate
                break
        if not file_path:
            print(f"Warning: File not found {base_name}(.docx/.png) in train folder. Skipping.")
            continue

        text = extract_text(file_path)
        if not text:
            continue
        doc = nlp(text)
        entities = []
        matcher = Matcher(nlp.vocab)
        # --- NEW MATCHER LOGIC ---
        for col_name, label in config.COLUMN_LABEL_MAP.items():
            value = row.get(col_name)
            if pd.notna(value):
                value_str = str(value)
                pattern = [{"TEXT": token} for token in value_str.split()]
                matcher.add(label, [pattern])
        matches = matcher(doc)
        # Convert token-level matches to character-level spans
        for match_id, start_token, end_token in matches:
            label_str = nlp.vocab.strings[match_id]
            span = doc[start_token:end_token]
            entities.append((span.start_char, span.end_char, label_str))
        if entities:
            training_data.append({'text': text, 'entities': list(set(entities))}) # Use set to remove duplicate matches
        else:
            print(f"Could not find any entities for: {row['file_name']}")
    print(f"Successfully created {len(training_data)} training examples.")
    return training_data


    print(f"Created {len(training_data)} training examples.")
    return training_data
