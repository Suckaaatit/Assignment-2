# Assignment-2

# 🧾 Metadata Extraction from Legal Documents

This project builds an ML pipeline to automatically extract structured metadata from legal documents, regardless of their layout or template. It supports both `.docx` files and scanned `.png` images as input.

---

## 📌 What It Extracts

The system extracts the following metadata fields:

- 📅 **Agreement Start Date**  
- 📅 **Agreement End Date**  
- 🔁 **Renewal Notice (Days)**  
- 👤 **Party One**  
- 👤 **Party Two**

---

## ⚙️ How It Works

1. **Text Extraction**  
   - `.docx`: Extracted using Python’s `docx` module  
   - `.png`: Handled via Tesseract OCR

2. **NER Model (spaCy)**  
   - A custom **Named Entity Recognition** model trained from scratch using the data you provide  
   - No regex or rule-based methods – fully ML-driven

3. **Evaluation**  
   - Outputs recall per field to evaluate model performance

---

## 🚀 Getting Started

### 1. 🧠 Install Tesseract OCR

- **Windows**: [Download here](https://github.com/tesseract-ocr/tesseract) and add to PATH  
- **macOS**:  
  ```bash
  brew install tesseract
Linux:

bash
Copy
Edit
sudo apt-get install tesseract-ocr
2. 🐍 Python Setup
bash
Copy
Edit
# Create virtual environment (recommended)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
🧪 How to Run
1. ✅ Prepare Data
Ensure these are correctly placed:

Training files: data/train/

Test files: data/test/

Annotations: data/train.csv, data/test.csv

2. 🏋️ Train the Model
bash
Copy
Edit
python -m src.train
3. 🔍 Predict & Evaluate
bash
Copy
Edit
python -m src.predict
This will print recall scores per field and show predicted metadata.

🌐 API (Optional)
Use your trained model as a web API with FastAPI.

1. Install API Dependencies
bash
Copy
Edit
pip install fastapi uvicorn python-multipart
2. Start the Server
bash
Copy
Edit
uvicorn src.api:app --reload
3. Access the Docs
Open in browser:
👉 http://127.0.0.1:8000/docs

You can upload .docx or .png files via /extract-metadata/ and receive JSON-formatted metadata.

Or, use curl:

bash
Copy
Edit
curl -X POST "http://127.0.0.1:8000/extract-metadata/" \
  -F "file=@yourfile.docx"
📈 Evaluation Metric
Recall is used to evaluate each metadata field:

ini
Copy
Edit
Recall = Correct Predictions / Total Relevant Instances

RESULTS:
Training the model:
<img width="391" height="364" alt="image" src="https://github.com/user-attachments/assets/9f33dd82-865e-4b45-9114-168ca023e4cb" />
<img width="397" height="390" alt="image" src="https://github.com/user-attachments/assets/0906b793-64fa-45e7-ad7c-d83861693479" />

Prediction:
<img width="281" height="83" alt="image" src="https://github.com/user-attachments/assets/1713b7a2-05a4-4c03-b85c-3b7dd87189ba" />



🧩 Built With
spaCy – NLP & NER engine

Tesseract OCR – Image text extraction

FastAPI – API interface

Python packages: docx, pandas, uvicorn, scikit-learn, and more

👥 Contributing
PRs are welcome! Please make sure your changes are well-tested and follow project conventions.

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
