# 🔤 Predicting_Next_Word

This project implements a **Next Word Prediction** model using **LSTM (Long Short-Term Memory)** networks trained on a sample text corpus. It allows users to input a sequence of words and predict the most likely next word using a trained deep learning model. The application is deployed using **Streamlit**.

---

## 🧠 Project Highlights

- Trained on **Shakespeare's Hamlet** (sample dataset: `hamlet.txt`)
- Utilizes **LSTM architecture** for sequential text prediction
- Includes **tokenizer** for text preprocessing
- Deployed using **Streamlit** with real-time predictions
- Robust prediction pipeline using `tokenizer.pickle` and `next_word_lstm.h5`

---

## 📁 Project Structure

```

.
├── app.py                  # Streamlit app for predicting next word
├── experiments.ipynb       # Training, tuning, and evaluation notebook
├── hamlet.txt              # Training corpus (Shakespeare's Hamlet)
├── next\_word\_lstm.h5       # Trained LSTM model
├── tokenizer.pickle        # Tokenizer used for text preprocessing
├── README.md               # Project documentation

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Predicting_Next_Word.git
cd Predicting_Next_Word
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

(If `requirements.txt` is not available, manually install: `tensorflow`, `streamlit`, `numpy`, `pandas`)

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📦 How It Works

1. User enters a sequence of words.
2. The input is tokenized and padded.
3. The LSTM model (`next_word_lstm.h5`) predicts the next word's token.
4. Token is converted back to a word using the `tokenizer.pickle`.
5. The predicted word is shown on the interface.

---

## 🧪 Model Training

Training and experiments are documented in `experiments.ipynb`. The training flow involves:

* Loading text corpus (`hamlet.txt`)
* Tokenizing sequences
* Building padded sequences
* Training an LSTM model with early stopping

---

## 📚 Dataset

* File: `hamlet.txt`
* Source: Public domain literary text
* Preprocessed and tokenized for LSTM input

---

## ⚙️ Environment

* Python 3.8+
* TensorFlow 2.x
* Streamlit
* NumPy, Pandas

Note: The app disables GPU/Metal acceleration by design for compatibility.

---

## 🌐 Deployment

This project is fully compatible with:

* **Streamlit Cloud**
* **Heroku**
* **Local deployment**

Just make sure to include all required files and environment specs (`requirements.txt`, optionally `runtime.txt`).

---

