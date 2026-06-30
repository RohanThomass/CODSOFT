💳 Credit Card Fraud Detection System

A Machine Learning-powered Flask web application that detects fraudulent credit card transactions using a **Random Forest Classifier**.

The application supports **batch prediction using CSV uploads**, displays analytics, and allows users to download prediction results.

---

🌐 Live Demo

**Render Deployment**

[https://YOUR-RENDER-URL.onrender.com](https://codsoft-ohht.onrender.com)

---

🚀 Features

- ✅ Batch prediction using CSV upload
- ✅ Fraud/Genuine transaction classification
- ✅ Download prediction results as CSV
- ✅ Interactive analytics dashboard
- ✅ Pie Chart & Bar Chart visualization
- ✅ Confusion Matrix (when actual labels are available)
- ✅ Automatic evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC

---

🛠 Tech Stack

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

```text
Credit_Card_Fraud_Detection/
│
├── app.py
├── train.py
├── requirements.txt
├── Procfile
├── runtime.txt
│
├── models/
│   ├── fraud_detector.pkl
│   └── scaler.pkl
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── static/
│   ├── css/
│   └── images/
│
├── uploads/
├── outputs/
│
└── data/
    └── raw/
```

---

🤖 Machine Learning Pipeline

1. Data Cleaning
2. Feature Scaling
3. Train-Test Split
4. SMOTE (Class Imbalance Handling)
5. Random Forest Training
6. Model Evaluation
7. Model Serialization (Joblib)
8. Flask Deployment

---

📊 Model Performance

| Metric | Score |
|---------|------:|
| Accuracy | Add your value |
| Precision | Add your value |
| Recall | Add your value |
| F1 Score | Add your value |
| ROC-AUC | Add your value |

---

#📁 Dataset

The original dataset is **not included** because it exceeds GitHub's file size limit.

Download it from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


⚙ Installation

Clone the repository:

```bash
git clone https://github.com/RohanThomass/CODSOFT.git
```

Go to the project folder:

```bash
cd CODSOFT/Credit_Card_Fraud_Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---



📌 Future Improvements

- User Authentication
- Docker Support
- Cloud Storage for Uploaded Files
- REST API
- Model Monitoring
- Explainable AI (SHAP)

---

👨‍💻 Author
Rohan Battepati

**Rohan Battepati**

Data Science Intern – CodSoft

GitHub:
https://github.com/RohanThomass
