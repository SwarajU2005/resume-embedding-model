
# 🧠 Resume Embedding Model with GTE-base

This project contains the training pipeline, fine-tuned model, and inference tools for building a **resume similarity and job-matching system** using [GTE-base](https://huggingface.co/thenlper/gte-base) as the backbone.

---

## 🚀 Features

- Fine-tuned sentence embedding model for resumes
- Supports resume ↔ job description matching
- Trained on real and synthetic resume datasets
- Smart chunking of long resumes for accurate semantic capture
- Inference script to rank resumes by query
- Integrated with Hugging Face Hub and model cards

---

## 📚 Model Card

- **Base model**: `thenlper/gte-base`
- **Fine-tuned model**: [`swaraj20/resume_gte_embedding`](https://huggingface.co/swaraj20/resume_gte_embedding)
- **Loss**: CosineSimilarityLoss
- **Training size**: 4,169 resume chunks
- **Use case**: Resume similarity, retrieval, job description matching

---

## 🧪 Example Usage

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("swaraj20/resume_gte_embedding")

resume = "Skilled in Python and cloud deployment. 2 years of backend engineering."
jd = "Looking for a backend engineer with Python and AWS experience."

emb1 = model.encode(resume, convert_to_tensor=True)
emb2 = model.encode(jd, convert_to_tensor=True)

sim = util.cos_sim(emb1, emb2)
print(f"Similarity: {sim.item():.4f}")
```

---

## 📂 Folders

- `train/` – fine-tuning script with smart chunking and multiple loss strategies
- `inference/` – search script to match resumes to input queries
- `model/` – zipped fine-tuned model (or Hugging Face link)
- `examples/` – demo resume files

---

## 🧠 How It Works

1. Resumes are loaded and chunked based on section headers (e.g., Education, Experience)
2. Three training strategies are used:
   - Self-supervised resume-resume pairs
   - Category-based resume pairs (same job role)
   - Resume ↔ synthetic job description pairs
3. Embeddings are fine-tuned using cosine similarity loss
4. Final model is saved to Hugging Face Hub

---

## 🧪 Test Your Query Against Resumes

```bash
python inference/infer_resumes.py --folder ./resumes --query "Looking for data analyst skilled in SQL and Excel"
```

---

## 📦 Installation

```bash
pip install -U sentence-transformers PyMuPDF python-docx torch
```

---

## 📝 License

MIT License – free to use, modify, and distribute with credit.
---

## ✨ Credits

Created by [@swaraj20](https://huggingface.co/swaraj20). Powered by 🤗 Hugging Face and Sentence Transformers.
