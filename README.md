# 🛡️ Fraudulent Transaction Detection 

A Machine Learning–powered web application built with **Streamlit** to detect fraudulent credit card transactions. This project uses a trained **Logistic Regression** model on the popular European credit card transactions dataset and provides an interactive UI for real-time fraud analysis.

---

## 📌 Project Overview

Credit card fraud detection is a critical problem due to highly imbalanced data and costly false negatives. This project addresses that challenge by:

* Training a classification model on anonymized transaction data
* Handling class imbalance using **under-sampling**
* Deploying the trained model as an interactive **Streamlit web app**
* Supporting multiple input methods (manual, CSV batch, example scenarios)

**Key Highlights**

* Dataset: European Credit Card Transactions
* Features: `Time`, `V1`–`V28` (PCA-transformed), `Amount`
* Model: Logistic Regression
* Performance: ~94% accuracy on test data
* Deployment: Streamlit Cloud

---

## 🚀 Live Demo

🔗 **Streamlit App**:
https://credit-card-fraud-detection-r5rb73b8zdnszbymbrappw2.streamlit.app/
---

## 🗂️ Repository Structure

```
Fraudulent-Transaction-Detection/
│
├── Project_Credit_Card_Fraud_Detection.ipynb   # EDA, preprocessing, training & evaluation
├── app.py                                     # Streamlit web application
├── fraud_model.pkl                            # Trained ML model (pickle file)
├── requirements.txt                           # Python dependencies
├── README.md                                  # Project documentation
```

---

## ⚙️ How It Works

### 🔹 Model Workflow

1. **Data Loading** – Load transaction CSV dataset
2. **Preprocessing** – Scale `Amount` and handle class imbalance
3. **Training** – Train Logistic Regression on balanced data
4. **Evaluation** – Measure accuracy, precision, and recall
5. **Deployment** – Serialize model and load it into Streamlit

---

## 🖥️ Application Features

* ✅ Example transaction scenarios (Legit / Fraud)
* 📂 Batch prediction using CSV upload
* ✍️ Manual transaction entry (30 features)
* 📊 Fraud summary metrics
* ⬇️ Downloadable prediction report
* 🎨 Clean and professional UI

---

## 🛠️ Tech Stack

* **Programming Language**: Python
* **ML Libraries**: Scikit-learn, NumPy, Pandas
* **Visualization**: Matplotlib
* **Web Framework**: Streamlit
* **Model Serialization**: Pickle

---

## 📦 Installation & Running Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Fraudulent-Transaction-Detection.git
cd Fraudulent-Transaction-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser.

---

## 📊 Input Format

The model expects **30 numerical features** in the following order:

```
Time, V1, V2, ..., V28, Amount
```

Ensure your CSV or manual input strictly follows this format.

---




## ⭐ Acknowledgements

* UCI Machine Learning Repository
* Streamlit Community
* Scikit-learn Documentation

---

If you like this project, don’t forget to ⭐ the repository!

