import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from collections import Counter
# from sklearn.model_selection import GridSearchCV
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


# param_grid = {
#     'max_depth': [3, 4, 5, 6, 7, 8, 9, None]  # Список значень max_depth для перевірки  [usually 3-10]
# }
#
# grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)  # cv - кількість фолдів
# grid_search.fit(X_train, y_train)
#
# best_max_depth = grid_search.best_params_['max_depth']
# print(f"Найкраща глибина дерева: {best_max_depth}")


# Побудова теплової карти кореляції Спірмена
f, ax = plt.subplots(figsize=(30, 25))  # Розмір графіку

# Обчислення кореляційної матриці Спірмена
mat = data.corr(method='spearman')

# Створення маски для верхньої частини теплової карти (щоб не дублювати інформацію)
mask = np.triu(np.ones_like(mat, dtype=bool))

# Вибір кольорової палітри
cmap = sns.diverging_palette(230, 20, as_cmap=True)  # Кольорова палітра, що розходиться

# Побудова теплової карти
sns.heatmap(mat,
            mask=mask,             # Застосування маски
            cmap=cmap,             # Використання обраної палітри
            vmax=1,                # Максимальне значення для кольорової шкали
            center=0,              # Центрування кольорової шкали на 0
            square=True,           # Квадратні клітинки
            linewidths=.5,       # Товщина ліній між клітинками
            cbar_kws={"shrink": .5} # Розмір кольорової шкали
           )

plt.title('Correlation Heatmap', fontsize=35)  # Заголовок графіку
plt.tight_layout()                             # Автоматичне коригування розмірів, щоб написи не зрізались

# Змінюємо колір фону графіка та області побудови графіку
plt.gcf().patch.set_facecolor('darkgrey')       # Фон всього графіка (figure)
plt.gca().set_facecolor('darkgrey')       # Фон області з графіком (axes)

plt.show()


def get_top_correlated(correlation_matrix, top_n=10):
    """
    Повертає топ-N найбільш корельованих пар показників.
    """
    # Перетворюємо матрицю в "довгий" формат
    corr_df = correlation_matrix.unstack().sort_values(ascending=False)
    # Видаляємо діагональні елементи та дублікати
    corr_df = corr_df[corr_df != 1]
    corr_df = corr_df.groupby(level=0).head(top_n)
    return corr_df

top_correlations = get_top_correlated(mat, top_n=10)
print(top_correlations)








# Створюємо модель Random Forest
# rf_model = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)  # f1 = 30
# rf_model = RandomForestClassifier(n_estimators=500, class_weight={0: 1, 1: 20}, random_state=42)
# rf_model.fit(X_resampled, y_resampled)  # added


# rf_model = BalancedBaggingClassifier(
#     n_estimators=500,
#     sampling_strategy="auto",  # Автоматичне балансування
#     replacement=False,  # Без повторення при вибірці
#     random_state=42
# )
#
#
# rf_model.fit(X_resampled, y_resampled)

# Якщо модель навчалась на DataFrame, перевіримо X_test:
# if isinstance(X_resampled, pd.DataFrame):
#     X_test = pd.DataFrame(X_test, columns=X_resampled.columns)


# Прогнозування ймовірностей
# y_probs = rf_model.predict_proba(X_test)[:, 1]  # Ймовірність класу 1
#
# # Прогнозування
# # y_pred = rf_model.predict(X_test)
# threshold = 0.2
# y_pred = (y_probs > threshold).astype(int)










# -----------------------------------------------------------------------------------

# Створення та навчання моделі логістичної регресії
# model = LogisticRegression(max_iter=500, class_weight={0: 1, 1: 3})  # 100 по дефолту
# model.fit(X_train, y_train)

# Прогнозування на тестовій вибірці
# y_pred = model.predict(X_test)
#
# # Оцінка якості моделі
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Точність моделі: {accuracy}")
# print(classification_report(y_test, y_pred))




# при рандом форест
# from sklearn.model_selection import RandomizedSearchCV
# import numpy as np
#
# param_dist = {
#     'class_weight': [{0: 1, 1: w} for w in np.arange(1, 10, 1)]
# }
#
# rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
# random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, scoring='f1', n_iter=5, cv=5)
# random_search.fit(X_train, y_train)
#
# print("Найкращий class_weight:", random_search.best_params_)

# result = 7
