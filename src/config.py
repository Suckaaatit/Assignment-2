# src/config.py

# Define paths
DATA_PATH = "data/"
TRAIN_DATA_PATH = DATA_PATH + "train.csv"
TEST_DATA_PATH = DATA_PATH + "test.csv"
TRAIN_FILES_PATH = DATA_PATH + "train/"
TEST_FILES_PATH = DATA_PATH + "test/"

# Path to save or load the trained model
MODEL_PATH = "models/ner_model"

# Define the labels for NER
LABELS = [
    "AGREEMENT_START_DATE",
    "AGREEMENT_END_DATE",
    "RENEWAL_NOTICE_DAYS",
    "PARTY_ONE",
    "PARTY_TWO",
]

# Mapping from CSV columns to NER labels
COLUMN_LABEL_MAP = {
    "agreement_start_date": "AGREEMENT_START_DATE",
    "agreement_end_date": "AGREEMENT_END_DATE",
    "renewal_notice_days": "RENEWAL_NOTICE_DAYS",
    "party_one": "PARTY_ONE",
    "party_two": "PARTY_TWO",
}

# Training configuration
N_ITERATIONS = 50 # Number of training loops
