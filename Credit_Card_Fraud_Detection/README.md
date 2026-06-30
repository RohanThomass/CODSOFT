# 💳 Credit Card Fraud Detection System

A Machine Learning web application that detects fraudulent credit card transactions using a Random Forest Classifier.

---

## 🚀 Features

- Upload CSV file
- Predict fraudulent transactions
- Download prediction results
- Interactive Dashboard
- Pie Chart & Bar Chart
- Confusion Matrix
- Model Evaluation (Accuracy, Precision, Recall, F1 Score, ROC-AUC)

---

## 🛠 Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Project Structure

```
Credit_Card_Fraud_Detection
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│       └── creditcard.csv
│
├── models/
│   ├── fraud_detector.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── predictions.csv
│
├── static/
│   ├── css/
│   └── images/
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── uploads/
```

---

## 📊 Machine Learning Workflow

1. Data Preprocessing
2. Feature Scaling
3. SMOTE for Class Imbalance
4. Random Forest Training
5. Model Evaluation
6. Flask Deployment

---

## 📈 Performance Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/RohanThomass/Credit_Card_Fraud_Detection.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train.py
```

### Run Application

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 📷 Screenshots

(Add screenshots here)

---

## 👨‍💻 Author

Rohan Battepati

CodSoft Data Science Internship