# TranquilAnswers

#  TranquilAnswers: AI-Powered Mental Health FAQ Assistant

*Find peace of mind through intelligent answers*

**TranquilAnswers** is an AI-driven semantic search tool designed to help users find trusted, meaningful responses to common mental health questions — instantly, privately, and compassionately.

By combining powerful **Sentence Transformers** with **FAISS** vector search, and a sleek **Gradio UI**, TranquilAnswers provides a safe, fast, and intelligent way to access reliable mental health information.

---

## 🧠 Built With

* **Python** – Core language
* **Sentence Transformers** – Semantic embeddings for question understanding
* **FAISS (Facebook AI Similarity Search)** – Efficient vector similarity search
* **Pandas** – Lightweight FAQ dataset handling
* **Gradio** – Browser-based conversational interface
* **NLP** – For intent-aware Q\&A

---

## 🎯 Goals

* Make mental health guidance accessible using conversational AI
* Deliver personalized answers via semantic search, not keyword matching
* Empower users to self-educate while preserving their privacy
* Open-source, modifiable, deployable anywhere

---

## 📸 Preview

![TranquilAnswers UI](https://user-images.githubusercontent.com/demo/tranquil-ui.png)

---

## 📁 Project Structure

```
TranquilAnswers/
├── app.py                 # Gradio-based chatbot logic
├── faqs.csv               # FAQ dataset (question, answer)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## ⚙️ Setup Instructions

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/TranquilAnswers.git
cd TranquilAnswers
```

### ✅ Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install sentence-transformers faiss-cpu pandas gradio
```

---

## 🧾 FAQ Dataset Format (`faqs.csv`)

```csv
question,answer
"What is anxiety?","Anxiety is a normal emotion but becomes a disorder when persistent or overwhelming."
"How can I deal with depression?","Reach out to a trusted person, maintain a routine, and consider professional help."
"What are symptoms of burnout?","Exhaustion, detachment, and reduced performance are common signs of burnout."
```

---

## 🖥️ Run the App

```bash
python app.py
```

Once running, it will open a browser at `http://localhost:7860` — ready for queries.

---

## 💬 Example Questions

* "What is CBT and how does it help?"
* "How can I manage panic attacks on my own?"
* "Is it normal to feel anxious every day?"
* "What’s the difference between stress and anxiety?"

---

## 🧠 `app.py` (Semantic Search + Gradio)

```python
import pandas as pd
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset
df = pd.read_csv("faqs.csv")  # Make sure it has 'question' and 'answer' columns
questions = df["question"].tolist()
answers = df["answer"].tolist()

# Encode questions
question_embeddings = model.encode(questions, convert_to_numpy=True)

# Create FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Response generator
def get_response(user_input):
    user_embedding = model.encode([user_input], convert_to_numpy=True)
    distance, idx = index.search(user_embedding, 1)
    if distance[0][0] < 1.0:
        return answers[idx[0][0]]
    else:
        return "I'm sorry, I couldn't find a good match. Try rephrasing your question."

# Launch Gradio interface
gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(lines=3, placeholder="Ask your mental health question..."),
    outputs=gr.Textbox(label="TranquilAnswers Response"),
    title="🧠 TranquilAnswers – Mental Health FAQ Assistant",
    description="Semantic FAQ assistant trained to help answer your mental health concerns in a private, conversational format."
).launch()
```

---

## 📦 `requirements.txt`

```txt
sentence-transformers
faiss-cpu
pandas
gradio
```

---

## 🔐 Privacy First

TranquilAnswers runs **locally**, and **no personal data is stored or sent online**. This makes it ideal for private self-help and confidential usage.

---

## 🔮 Future Plans

* [ ] Voice input (with Whisper) + TTS
* [ ] Deploy to Hugging Face Spaces
* [ ] Add multilingual support
* [ ] Integrate a journaling or mood tracker
* [ ] Expand to contextual conversation memory

---

## 🙋‍♂️ Author

**Varun Haridas**
📧 [varun.haridas321@gmail.com](mailto:varun.haridas321@gmail.com)

> Built with ❤️ to help people find clarity, calm, and comfort through AI.

---

## ⭐️ Support This Project

If TranquilAnswers helped you or inspired you, consider starring the repo on GitHub and sharing it with others.

---
