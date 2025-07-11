from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from src import config
from src.data_handler import extract_text
import spacy

app = FastAPI()

# Load the trained spaCy NER model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/ner_model')
nlp = spacy.load(MODEL_PATH)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '../tmp_uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract-metadata/")
def extract_metadata(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_ext = os.path.splitext(file.filename)[1]
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text from the uploaded file
    text = extract_text(temp_path)
    doc = nlp(text)

    # Prepare response for required fields
    result = {label.lower(): None for label in config.LABELS}
    for ent in doc.ents:
        if ent.label_ in config.LABELS:
            # Only take first occurrence for each label
            if result[ent.label_.lower()] is None:
                result[ent.label_.lower()] = ent.text

    # Clean up temp file
    os.remove(temp_path)

    return JSONResponse({"extracted_metadata": result})

# To run the API:
# uvicorn src.api:app --reload
