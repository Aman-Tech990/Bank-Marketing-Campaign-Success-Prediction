import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =========================
# CLEAN MODEL TRAINING
# =========================

df = pd.read_csv("bank.csv")

# Select only relevant columns
df = df[[
    "age",
    "balance",
    "campaign",
    "housing",
    "poutcome",
    "deposit"
]]

# Encode categorical features manually
df["housing"] = df["housing"].map({"yes": 1, "no": 0})

df["poutcome"] = df["poutcome"].map({
    "success": 2,
    "failure": 1,
    "other": 0,
    "unknown": 0
})

df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

X = df.drop("deposit", axis=1)
y = df["deposit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# GUI START
# =========================

root = tk.Tk()
root.title("AI Deposit Decision Engine")
root.geometry("1000x600")
root.configure(bg="#F1F5F9")

container = tk.Frame(root, bg="#F1F5F9")
container.pack(fill="both", expand=True, padx=40, pady=30)

# =========================
# LEFT PANEL
# =========================

left_panel = tk.Frame(container, bg="white")
left_panel.pack(side="left", fill="both", expand=True, padx=(0, 20))

tk.Label(
    left_panel,
    text="Customer Profile",
    font=("Segoe UI", 20, "bold"),
    bg="white",
    fg="#1E293B"
).pack(pady=(30, 20))

form = tk.Frame(left_panel, bg="white")
form.pack()

def label(text, row):
    tk.Label(form, text=text,
             font=("Segoe UI", 11),
             bg="white",
             fg="#475569").grid(row=row, column=0, sticky="w", pady=12)

def entry(row):
    e = ttk.Entry(form, width=28)
    e.grid(row=row, column=1, pady=12, padx=20)
    return e

label("Age", 0)
age_entry = entry(0)

label("Account Balance (₹)", 1)
balance_entry = entry(1)

label("Campaign Contacts", 2)
campaign_entry = entry(2)

label("Housing Loan", 3)
housing_var = tk.StringVar()
housing_dropdown = ttk.Combobox(
    form,
    textvariable=housing_var,
    values=["no", "yes"],
    state="readonly",
    width=25
)
housing_dropdown.grid(row=3, column=1, pady=12)
housing_dropdown.current(0)

label("Previous Campaign Outcome", 4)
poutcome_var = tk.StringVar()
poutcome_dropdown = ttk.Combobox(
    form,
    textvariable=poutcome_var,
    values=["success", "failure", "other", "unknown"],
    state="readonly",
    width=25
)
poutcome_dropdown.grid(row=4, column=1, pady=12)
poutcome_dropdown.current(3)

# =========================
# RIGHT PANEL
# =========================

right_panel = tk.Frame(container, bg="#1E3A8A")
right_panel.pack(side="right", fill="both", expand=True)

result_title = tk.Label(
    right_panel,
    text="Prediction Result",
    font=("Segoe UI", 20, "bold"),
    bg="#1E3A8A",
    fg="white"
)
result_title.pack(pady=40)

percentage_label = tk.Label(
    right_panel,
    text="-- %",
    font=("Segoe UI", 50, "bold"),
    bg="#1E3A8A",
    fg="white"
)
percentage_label.pack()

decision_label = tk.Label(
    right_panel,
    text="Awaiting Analysis",
    font=("Segoe UI", 16),
    bg="#1E3A8A",
    fg="white"
)
decision_label.pack(pady=20)

confidence_bar = ttk.Progressbar(
    right_panel,
    orient="horizontal",
    length=300,
    mode="determinate"
)
confidence_bar.pack(pady=20)

# =========================
# PREDICT FUNCTION
# =========================

def predict():
    try:
        user_data = pd.DataFrame([{
            "age": int(age_entry.get()),
            "balance": float(balance_entry.get()),
            "campaign": int(campaign_entry.get()),
            "housing": 1 if housing_var.get() == "yes" else 0,
            "poutcome": {
                "success": 2,
                "failure": 1,
                "other": 0,
                "unknown": 0
            }[poutcome_var.get()]
        }])

        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0][1] * 100

        confidence_bar["value"] = probability
        percentage_label.config(text=f"{probability:.1f}%")

        if prediction == 1:
            right_panel.config(bg="#16A34A")
            percentage_label.config(bg="#16A34A")
            decision_label.config(
                text="LIKELY TO SUBSCRIBE",
                bg="#16A34A"
            )
            result_title.config(bg="#16A34A")
        else:
            right_panel.config(bg="#DC2626")
            percentage_label.config(bg="#DC2626")
            decision_label.config(
                text="NOT LIKELY TO SUBSCRIBE",
                bg="#DC2626"
            )
            result_title.config(bg="#DC2626")

    except:
        messagebox.showerror("Input Error", "Please enter valid values.")

# =========================
# BUTTON
# =========================

analyze_btn = tk.Button(
    left_panel,
    text="Analyze Customer",
    command=predict,
    font=("Segoe UI", 12, "bold"),
    bg="#1E3A8A",
    fg="white",
    padx=25,
    pady=10,
    bd=0
)
analyze_btn.pack(pady=30)

root.mainloop()

