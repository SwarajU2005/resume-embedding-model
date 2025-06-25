
# ✅ Import required libraries
import os
import re
import textwrap
import fitz  # PyMuPDF
import pandas as pd
import json
from docx import Document
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

# ✅ Define smart chunking function
MAX_CHUNK_CHARS = 1500
SECTION_HEADINGS = [
    "Education", "Skills", "Work Experience", "Experience",
    "Projects", "Certifications", "Objective", "Summary",
    "Achievements", "Personal Information"
]

def smart_chunk_resume(text):
    text = re.sub(r'\s+', ' ', text.strip())
    pattern = r'(?i)(' + '|'.join(re.escape(h) for h in SECTION_HEADINGS) + r')[:\n]'
    parts = re.split(pattern, text)

    chunks = []
    i = 0
    while i < len(parts):
        if parts[i].strip().lower() in [h.lower() for h in SECTION_HEADINGS]:
            heading = parts[i].strip()
            body = parts[i + 1].strip() if (i + 1) < len(parts) else ""
            combined = f"{heading}: {body}"
            chunks.extend(textwrap.wrap(combined, MAX_CHUNK_CHARS))
            i += 2
        else:
            unstructured = parts[i].strip()
            if len(unstructured) > 0:
                chunks.extend(textwrap.wrap(unstructured, MAX_CHUNK_CHARS))
            i += 1
    return chunks

# ✅ Load files and extract + chunk text
texts = []
df = None

root_folder = "/content/resume_folder"  # Replace if needed

for root, dirs, files in os.walk(root_folder):
    for file in files:
        path = os.path.join(root, file)

        try:
            if file.endswith(".pdf"):
                doc = fitz.open(path)
                text = "\n".join([page.get_text() for page in doc])
                doc.close()
                chunks = smart_chunk_resume(text)
                texts.extend(chunks)

            elif file.endswith(".docx"):
                doc = Document(path)
                text = "\n".join([para.text for para in doc.paragraphs])
                chunks = smart_chunk_resume(text)
                texts.extend(chunks)

            elif file.endswith(".csv"):
                df = pd.read_csv(path)
                df.columns = df.columns.str.lower()  # normalize column names
                if 'resume' in df.columns and 'category' in df.columns:
                    for _, row in df.iterrows():
                        resume = str(row['resume'])
                        chunks = smart_chunk_resume(resume)
                        texts.extend(chunks)

            elif file.endswith(".json"):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            chunks = smart_chunk_resume(str(item))
                            texts.extend(chunks)
                    elif isinstance(data, dict):
                        chunks = smart_chunk_resume(str(data))
                        texts.extend(chunks)

        except Exception as e:
            print(f"Error processing {file}: {e}")

print(f"✅ Total text chunks: {len(texts)}")

# ✅ Create InputExamples (self-supervised)
self_supervised_examples = []
for i in range(0, len(texts) - 1, 2):
    if len(texts[i]) > 30 and len(texts[i+1]) > 30:
        self_supervised_examples.append(InputExample(texts=[texts[i], texts[i+1]], label=0.9))

# ✅ Create category-based and JD-based examples (if CSV is present)
category_examples, jobdesc_examples = [], []
if df is not None:
    try:
        df.columns = df.columns.str.lower()
        from collections import defaultdict
        grouped = defaultdict(list)
        for _, row in df.iterrows():
            resume = str(row['resume']).strip()
            category = str(row['category']).strip()
            if len(resume) > 30:
                grouped[category].append(resume)

        for category, resumes in grouped.items():
            for i in range(0, len(resumes) - 1, 2):
                category_examples.append(InputExample(texts=[resumes[i], resumes[i+1]], label=0.95))

        for _, row in df.iterrows():
            resume = str(row['resume']).strip()
            category = str(row['category']).strip()
            if len(resume) > 30:
                jd = f"We are looking for a skilled {category} to join our team. The ideal candidate should have expertise in industry-standard tools and techniques relevant to {category} roles."
                jobdesc_examples.append(InputExample(texts=[resume, jd], label=0.95))
    except KeyError as e:
        print(f"❌ CSV missing expected column: {e}")

# ✅ Combine all training examples
all_examples = self_supervised_examples + category_examples + jobdesc_examples
train_dataloader = DataLoader(all_examples, shuffle=True, batch_size=8)

# ✅ Load and fine-tune model
model = SentenceTransformer("thenlper/gte-base")
train_loss = losses.CosineSimilarityLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=6,
    warmup_steps=150,
    show_progress_bar=True
)

# ✅ Save and zip model
model.save("fine-tuned-resume-gte-final")
import shutil
shutil.make_archive("fine-tuned-resume-gte-final", 'zip', "fine-tuned-resume-gte-final")
print("✅ Final model saved and zipped.")
