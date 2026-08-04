# 📝 Text Summarizer

A Text Summarization web application built using **FastAPI** and **Hugging Face Transformers (T5)**. The application generates concise summaries from long text using a fine-tuned T5 model.

---

## 🚀 Features

- 🤖 AI-powered text summarization using T5
- ⚡ FastAPI backend for high performance
- 🎨 Simple and responsive web interface
- 🧹 Automatic text preprocessing
- 📄 Supports long text inputs
- 💻 Easy to run locally

---

## 🛠️ Tech Stack

- Python
- FastAPI
- Hugging Face Transformers
- PyTorch
- Jinja2
- HTML, CSS, JavaScript

---

## 📂 Project Structure

```
Text_Summarizer/
│
├── app.py
├── templates/
│   └── index.html
├── saved_summary_model/
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── generation_config.json
│   ├── model.safetensors
│   └── spiece.model
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/kartikrastogi18/Text_Summarizer.git
cd Text_Summarizer
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the application

```bash
uvicorn app:app --reload
```

Open your browser:

```
http://127.0.0.1:8000
```

---

## 📖 API Endpoint

### POST `/summarize`

Generates a summary for the given text.

#### Request

```json
{
  "dialogue": "Enter your text here..."
}
```

#### Response

```json
{
  "summary": "Generated summary..."
}
```

---

## 🧠 Model

This project uses a fine-tuned **T5 Transformer** model for abstractive text summarization.

Model loading:

```python
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
```

---

## ✨ Screenshots

Add screenshots of your application here.

```
assets/
    home.png
```

Example:

```markdown
![Home Page](assets/home.png)
```

---

## 📌 Future Improvements

- Upload PDF and summarize
- DOCX file summarization
- Adjustable summary length
- Multi-language support
- Export summary as PDF
- Dark mode UI
- Deploy on Render/Railway

---

## 👨‍💻 Author

**Kartik Rastogi**

- GitHub: https://github.com/kartikrastogi18
- LinkedIn: https://linkedin.com/in/kartikrastogi18

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
