# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

# 2. Create output directories
os.makedirs("outputs/confusion_matrices", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 3. Load Dataset
df = pd.read_csv("/Users/mostafamashhadizadeh/Desktop/MyProjects/Churn_Analysis_Project/data/telco.csv")

# 4. Clean Data
df = df[df["TotalCharges"] != " "]
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["Churn"] = df["Churn"].map({"Churned": 1, "Stayed": 0})
df.drop("customerID", axis=1, inplace=True)
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)

# 5. Feature/Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 6. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 8. Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 9. Train Models
models = {}

# --- Random Forest
print("\n--- Random Forest ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train_res, y_train_res)
rf_pred = rf_model.predict(X_test)
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))
models["Random Forest"] = (rf_model, rf_pred)

# --- XGBoost
print("\n--- XGBoost ---")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1.5, random_state=42)
xgb_model.fit(X_train_res, y_train_res)
xgb_pred = xgb_model.predict(X_test)
print(classification_report(y_test, xgb_pred))
print("Accuracy:", accuracy_score(y_test, xgb_pred))
models["XGBoost"] = (xgb_model, xgb_pred)

# --- SVM
print("\n--- SVM ---")
svm_model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
svm_model.fit(X_train_res, y_train_res)
svm_pred = svm_model.predict(X_test)
print(classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))
models["SVM"] = (svm_model, svm_pred)

# --- Logistic Regression
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_model.fit(X_train_res, y_train_res)
lr_pred = lr_model.predict(X_test)
print(classification_report(y_test, lr_pred))
print("Accuracy:", accuracy_score(y_test, lr_pred))
models["Logistic Regression"] = (lr_model, lr_pred)

# --- Naive Bayes
print("\n--- Naive Bayes ---")
nb_model = GaussianNB()
nb_model.fit(X_train_res, y_train_res)
nb_pred = nb_model.predict(X_test)
print(classification_report(y_test, nb_pred))
print("Accuracy:", accuracy_score(y_test, nb_pred))
models["Naive Bayes"] = (nb_model, nb_pred)

# --- Neural Network (MLPClassifier)
print("\n--- Neural Network (MLPClassifier) ---")
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_model.fit(X_train_res, y_train_res)
mlp_pred = mlp_model.predict(X_test)
print(classification_report(y_test, mlp_pred))
print("Accuracy:", accuracy_score(y_test, mlp_pred))
models["Neural Net (MLP)"] = (mlp_model, mlp_pred)

# 10. Confusion Matrices
for name, (model, y_pred) in models.items():
    plt.figure(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrices/{name.replace(' ', '_')}.png")
    plt.close()

# 11. Accuracy Comparison
model_scores = {name: accuracy_score(y_test, pred) for name, (_, pred) in models.items()}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()), palette='Set2')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=30)
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/accuracy_comparison.png")
plt.close()

# 12. ROC Curves
plt.figure(figsize=(8, 5))
for name, (model, _) in models.items():
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    except:
        continue

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/roc_curves.png")
plt.close()

# 13. Summary Table
summary = pd.DataFrame({
    'Model': list(models.keys()),
    'Accuracy': [accuracy_score(y_test, pred) for _, pred in models.values()]
})
auc_list = []
for name, (model, _) in models.items():
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
        else:
            auc_list.append(np.nan)
            continue
        auc_val = roc_auc_score(y_test, y_score)
        auc_list.append(auc_val)
    except:
        auc_list.append(np.nan)
summary["AUC"] = auc_list

print("\n🔍 Model Performance Summary:")
print(summary)

summary.to_csv("results/summary.csv", index=False)