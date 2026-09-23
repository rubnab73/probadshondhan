---

title: Probad Bangla Proverb AI

emoji: 📜

colorFrom: indigo

colorTo: blue

sdk: docker

app_port: 7860

pinned: false

---

# 📜 Probad — Bangla Situation-to-Proverb AI Retrieval Engine

> **An advanced hybrid AI search engine that analyzes real-life situations and metaphorical descriptions to retrieve the most relevant Bangla proverb.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Transformer: BanglaBERT](https://img.shields.io/badge/NLP-BanglaBERT-orange.svg)](https://huggingface.co/csebuetnlp/banglabert)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Key Features

### 🔹 Hybrid Neural + Lexical Ensemble

* **BanglaBERT (Pretrained Transformer):** Uses BUET's `csebuetnlp/banglabert` model to capture metaphorical meaning, semantic relationships, contextual similarities, and deeper interpretations in **768-dimensional embeddings**.
* **Scikit-Learn TF-IDF:** Provides fast lexical matching and keyword-level precision.
* **Score Fusion:** Combines **75% deep contextual similarity + 25% lexical similarity** to produce the final ranking.

### 🔹 Dual-Layer Augmented Retrieval

* Matches the user's situation against the **meaning/description of each proverb**.
* Also incorporates **real-world training situations** using k-nearest-neighbor (kNN) matching.
* This allows the system to connect both the explicit meaning of a proverb and examples of how that proverb applies in real situations.

### 🔹 Evaluation Performance

Based on a **25-example human-written real-world situation benchmark**:

* **Top-1 Accuracy:** 80.0%
* **Top-5 Accuracy:** 88.0%
* **MRR:** 0.8180
* **nDCG@5:** 0.8327

### 🔹 Modern Minimal Web Interface

* Dynamic search and filtering.
* Interactive benchmark leaderboard.
* Explorer for the **500-proverb knowledge base**.
* Clean and responsive user interface.

---

## 📊 Model Evaluation & Benchmark

The system was evaluated using a **blind human-gold test containing 25 real-world situation descriptions** written independently.

| Model / Algorithm              | Architecture                     | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy |     MRR    |   nDCG@5   |
| :----------------------------- | :------------------------------- | :------------: | :------------: | :------------: | :--------: | :--------: |
| **[Unified Proverb Engine] ★** | Hybrid Neural + Lexical Ensemble |    **80.0%**   |    **80.0%**   |    **88.0%**   | **0.8180** | **0.8327** |
| **TF-IDF Lexical**             | Scikit-Learn TF-IDF Engine       |      48.0%     |      64.0%     |      76.0%     |   0.5813   |   0.6256   |
| **BanglaBERT Contextual**      | Deep Contextual Transformer      |      24.0%     |      48.0%     |      60.0%     |   0.3727   |   0.4291   |

---

## 🧠 How the System Works

The retrieval pipeline follows a multi-stage architecture:

```text
User Situation
      │
      ▼
Bangla Text Preprocessing
      │
      ├──────────────────┐
      ▼                  ▼
   TF-IDF            BanglaBERT
   Retrieval         Embedding
      │                  │
      │                  ▼
      │             Semantic Similarity
      │
      ▼
Lexical Similarity
      │
      └──────────┬──────────┘
                 ▼
          Score Fusion
     75% BERT + 25% TF-IDF
                 │
                 ▼
        Final Ranked Results
                 │
                 ▼
        Most Relevant Proverb
```

The system therefore combines **lexical matching** with **contextual semantic understanding**, allowing it to retrieve proverbs even when the user's situation does not contain the exact words used in the proverb.

---

## 📁 Project Structure

```bash
Probad/

├── app.py                          # Flask web application server
├── requirements.txt                # Python dependencies
├── .gitignore                      # Dataset and cache protection
│
├── src/
│   ├── pipeline.py                 # Unified retrieval pipeline
│   ├── preprocessor.py             # Bangla text normalizer and tokenizer
│   ├── evaluate.py                 # Automated benchmark evaluation
│   │
│   └── models/
│       ├── banglabert_retriever.py # BanglaBERT neural retriever
│       ├── tfidf_retriever.py      # TF-IDF lexical retriever
│       └── unified_engine.py       # Hybrid retrieval engine
│
├── web/
│   ├── templates/
│   │   └── index.html              # Web user interface
│   │
│   └── static/
│       ├── css/
│       │   └── style.css           # Custom responsive styling
│       └── js/
│           └── app.js              # Client-side interactions
│
├── dataset/
│   ├── proverbs.csv                # 500 Bangla proverbs and meanings
│   └── situations_train.sample.csv # Training situation sample schema
│
└── results/
    ├── benchmark_results.json      # Cached benchmark results
    └── benchmark_leaderboard.md    # Detailed evaluation leaderboard
```

---

## 🚀 Local Setup & Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/probad.git

cd probad
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows:**

```powershell
.\.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 🔬 Technology Stack

| Component           | Technology                  |
| ------------------- | --------------------------- |
| Language            | Python 3.10+                |
| Web Framework       | Flask                       |
| Lexical Retrieval   | Scikit-Learn TF-IDF         |
| Semantic Retrieval  | BanglaBERT                  |
| Embedding Dimension | 768                         |
| Ranking             | Hybrid Score Fusion         |
| Semantic Weight     | 75%                         |
| Lexical Weight      | 25%                         |
| Dataset             | 500 Bangla Proverbs         |
| Evaluation Set      | 25 Human-Written Situations |

---

## 🎯 Project Goal

**Probad** aims to bridge the gap between natural-language situations and traditional Bangla proverbs.

Instead of requiring users to remember a proverb or provide its exact keywords, the system allows them to describe a **situation, experience, or metaphor in natural Bangla** and retrieves proverbs that convey a similar underlying meaning.

For example:

```text
Input:
"সে নিজের ভুলের জন্য অন্যকে দোষ দিচ্ছে,
যদিও সমস্যাটা আসলে তার নিজের তৈরি।"

        ↓

AI Retrieval

        ↓

Relevant Bangla Proverb
```

The core idea is to move from:

> **Situation → Meaning → Proverb**

rather than simply:

> **Keyword → Proverb**

---

## 📌 Current Scope

The current knowledge base contains **500 Bangla proverbs** with associated meanings and retrieval information.

The system currently supports:

* Natural-language Bangla situation queries
* Lexical similarity search
* Contextual semantic retrieval
* Hybrid score fusion
* Ranked proverb recommendations
* Benchmark evaluation
* Interactive web-based exploration

---

## 📜 License

This project is released under the **MIT License**.
