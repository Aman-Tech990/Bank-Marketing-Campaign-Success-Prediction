# Bank Marketing Campaign Success Prediction

A machine learning project that predicts whether a customer will subscribe to a bank term deposit based on demographic and marketing campaign attributes.

This project uses a Random Forest classifier to model marketing campaign success using structured banking data.

---

## Problem Statement

Banks conduct marketing campaigns to promote term deposits. However, not every customer subscribes.

The objective of this project is:

To predict whether a customer will subscribe to a term deposit using customer demographic and campaign-related features.

Marketing campaign success is defined as:

- `yes` → Customer subscribed  
- `no` → Customer did not subscribe  

---

## Dataset

Dataset: Bank Marketing Dataset  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset  

The dataset contains customer-related attributes such as:

- Age  
- Job  
- Marital status  
- Education  
- Account balance  
- Housing loan  
- Personal loan  
- Contact type  
- Campaign interactions  
- Previous campaign outcome  

Target variable:

- `deposit`

---

## Project Workflow

### 1. Data Loading

```python
import pandas as pd

df = pd.read_csv("bank.csv")
```

---

### 2. Feature and Target Separation

```python
y = df["deposit"]
X = df.drop("deposit", axis=1)
```

- `X` contains input features  
- `y` contains the target variable  

---

### 3. Encoding Categorical Variables

Several columns are categorical (job, marital, education, etc.).  
These are converted into numerical format using one-hot encoding:

```python
X = pd.get_dummies(X, drop_first=True)
```

This ensures the model can process categorical information effectively.

---

### 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

- 80% Training Data  
- 20% Testing Data  
- Stratified split maintains class balance  

---

### 5. Model Selection

Random Forest Classifier was chosen because:

- Works well on structured tabular datasets  
- Handles non-linear relationships  
- Reduces overfitting compared to a single decision tree  
- Provides strong baseline performance  

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
```

---

### 6. Model Training

```python
model.fit(X_train, y_train)
```

---

### 7. Model Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## Results

Test Accuracy: **85.9%**

Classification Report (Test Data):

| Class | Precision | Recall | F1-Score |
|-------|----------|--------|----------|
| No    | 0.89     | 0.83   | 0.86     |
| Yes   | 0.83     | 0.89   | 0.86     |

### Observations

- Balanced performance across both classes  
- Strong recall for predicting successful subscriptions  
- Slight overfitting observed (training accuracy higher than testing accuracy), which is common in ensemble models  

---

## Model Overview

Random Forest works by:

1. Building multiple decision trees on random subsets of data  
2. Training each tree independently  
3. Aggregating predictions using majority voting  

This ensemble approach improves generalization and reduces variance.

---

## Project Structure

```
├── AIML_project.ipynb
├── bank.csv
└── README.md
```

---

## How to Run

1. Clone the repository

```bash
git clone <your-repository-link>
```

2. Install dependencies

```bash
pip install pandas scikit-learn
```

3. Run the notebook

```bash
jupyter notebook AIML_project.ipynb
```

---

## Future Improvements

- Add ROC-AUC score evaluation  
- Visualize feature importance  
- Perform hyperparameter tuning using GridSearchCV  
- Save trained model using joblib  
- Deploy using Streamlit or Flask  

---

## Why This Project Matters

Marketing campaigns require significant resources. Predicting customer subscription likelihood can:

- Improve campaign targeting  
- Reduce marketing costs  
- Increase return on investment  
- Enable data-driven business decisions  

This project demonstrates the practical application of supervised machine learning in solving a real-world business problem.

---

## Team NeuroX
Aman Parida (2401020533)
Rohit Kumar Pradhan (2401020571)
Mantosa Kumar Biswal (2401020555)
Pratyush Beura (2401020560)
Rohan Sahoo (2401020570)
Chandra Shekhar Sahoo (2401020518)

B.Tech Computer Science (Data Science)
Group - 5(V)
Semester - 4th
