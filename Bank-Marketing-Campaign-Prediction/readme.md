# 🏦 Bank Marketing Campaign Success Prediction  
### AI-Powered Customer Conversion Intelligence
 
> Turning Blind Marketing into Data-Driven Targeting 

---

## 📌 Overview

Banks spend huge resources running marketing campaigns to promote term deposits.  
But not every customer converts.

This project builds a **Machine Learning Classification System** that predicts:

> 💬 *Will a customer subscribe to a term deposit?*

By using structured banking data and a Random Forest classifier, we convert a traditional marketing problem into a **data-driven decision system**.

---

## 🎯 Business Problem

Traditional Campaign Approach:

- 📞 Call thousands of customers
- 💰 Spend large operational cost
- 📉 Low conversion rate

Smart AI Approach:

- 🎯 Predict high-probability customers
- 📊 Target only promising leads
- 💵 Improve ROI

This project answers:

> “Can we predict subscription likelihood before contacting the customer?”

---

## 📂 Dataset Information

**Dataset:** Bank Marketing Dataset  
**Source:** Kaggle  
**Link:** https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset  

### 🔎 Features Used

- `age`
- `balance`
- `campaign`
- `housing`
- `poutcome`

### 🎯 Target Variable

- `deposit`
  - `yes` → Subscribed
  - `no` → Not Subscribed

---

## ⚙️ Machine Learning Pipeline

### 1️⃣ Data Loading

```python
import pandas as pd
df = pd.read_csv("bank.csv")
```

---

### 2️⃣ Feature Selection

```python
df = df[[
    "age",
    "balance",
    "campaign",
    "housing",
    "poutcome",
    "deposit"
]]
```

---

### 3️⃣ Encoding Categorical Features

```python
df["housing"] = df["housing"].map({"yes": 1, "no": 0})

df["poutcome"] = df["poutcome"].map({
    "success": 2,
    "failure": 1,
    "other": 0,
    "unknown": 0
})

df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
```

---

### 4️⃣ Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop("deposit", axis=1)
y = df["deposit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 5️⃣ Model Selection

We used **Random Forest Classifier** because:

- Works well on structured data  
- Handles non-linear relationships  
- Reduces overfitting  
- Provides stable performance  

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
```

---

### 6️⃣ Model Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 📈 Model Performance

- Test Accuracy: ~85–86%
- Balanced precision & recall
- Handles class imbalance using `class_weight="balanced"`

---

## 🖥️ GUI Application

A professional desktop GUI built using **Tkinter** allows users to:

- Enter customer details
- Analyze subscription probability
- View dynamic percentage output
- See clear decision result (Likely / Not Likely)

### Example High Probability Input

| Feature | Value |
|----------|--------|
| Age | 45 |
| Balance | 400000 |
| Campaign | 1 |
| Housing | No |
| Poutcome | Success |

### Example Low Probability Input

| Feature | Value |
|----------|--------|
| Age | 21 |
| Balance | 2000 |
| Campaign | 8 |
| Housing | Yes |
| Poutcome | Failure |

---

## 📁 Project Structure

```
📦 Bank-Marketing-Campaign-Success-Prediction
│
├── 📄 AIML_project.ipynb      # Model training notebook
├── 🐍 app.py                  # GUI application
├── 📊 bank.csv                # Dataset
├── 📘 README.md               # Project documentation
└── 📦 requirements.txt        # Dependencies
```

---

## ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Aman-Tech990/Bank-Marketing-Campaign-Success-Prediction
cd Bank-Marketing-Campaign-Success-Prediction
```

### 2️⃣ Install Dependencies

```bash
pip install pandas scikit-learn
```

### 3️⃣ Run Application

```bash
python app.py
```

---

## 🧠 Why Random Forest?

Random Forest:

1. Builds multiple decision trees  
2. Uses random subsets of data  
3. Combines predictions via majority voting  

This improves accuracy and reduces variance compared to a single decision tree.

---

## 🔧 Future Enhancements

- Add ROC-AUC curve visualization  
- Deploy using Streamlit / Flask  
- Add feature importance graph  
- Implement hyperparameter tuning  
- Deploy as cloud-hosted web app  

---

## 👥 Team NeuroX

- Aman Parida  
- Rohit Kumar Pradhan  
- Mantosa Kumar Biswal  
- Pratyush Beura  
- Rohan Sahoo  
- Chandra Shekhar Sahoo  

B.Tech Computer Science (Data Science)  
Semester – 4th  

---

## 💡 Real-World Impact

This system helps banks:

- 🎯 Improve targeting accuracy  
- 💰 Reduce marketing costs  
- 📊 Increase campaign ROI  
- 🧠 Make data-driven decisions  

Machine Learning turns guessing into intelligence.

---

## ⭐ If You Found This Useful

Give this repository a ⭐ and support the project!
