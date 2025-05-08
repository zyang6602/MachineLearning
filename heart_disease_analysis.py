import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('cardio_train.csv', sep=';')

print("Shape of the data:", df.shape)
df.head()

print(df.isnull().sum())

sns.countplot(x='cardio', data=df)
plt.title("Target Distribution (0 = No Disease, 1 = Disease)")
plt.show()

if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

X = df.drop('cardio', axis=1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"Accuracy for Logistic Regression: {accuracy_log:.4f}")

knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy for KNN: {accuracy_knn:.4f}")

plt.bar(['Logistic Regression', 'KNN'], [accuracy_log, accuracy_knn])
plt.ylim([0, 1])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()