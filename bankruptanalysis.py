import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Завантаження даних
data = pd.read_csv("data.csv")

# Розділення на ознаки (X) та цільову змінну (y)
X = data.drop("Bankrupt?", axis=1)
y = data["Bankrupt?"]

print("Розподіл класів перед обробкою:", Counter(y))

# Розділення на тренувальну та тестову вибірки (з урахуванням stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Аналіз розподілу класів у тренувальній вибірці
print("Розподіл класів у тренувальній вибірці перед обробкою:", Counter(y_train))


# Застосування SMOTE + Tomek Links для обробки незбалансованих даних
smote = SMOTE(random_state=42, sampling_strategy=0.3, k_neighbors=5)
tomek = TomekLinks()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
X_resampled, y_resampled = tomek.fit_resample(X_resampled, y_resampled)

print("Розподіл класів після обробки:", Counter(y_resampled))

rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
rf_model.fit(X_resampled, y_resampled)

# Прогнозування на тестовій вибірці
y_pred = rf_model.predict(X_test)

# Оцінка результатів
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність моделі: {accuracy}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Обчислення AUC-ROC
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Отримуємо ймовірності для класу "1"
auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC: {auc_roc}")

























