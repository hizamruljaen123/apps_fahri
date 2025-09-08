# SVM for Eligibility Classification of PIP Assistance (SMP Negeri)

## Introduction
The **Program Indonesia Pintar (PIP)** provides financial assistance to students from underprivileged families.  
To ensure fair distribution, we can use **Support Vector Machine (SVM)** to classify whether a student is **eligible** or **not eligible** based on socioeconomic and academic features.

## Features (Example)
- Family Income  
- Number of Dependents  
- Parents' Occupation  
- Student Attendance  
- Academic Performance (Grades)  

Label:  
- `1` = Eligible for PIP  
- `0` = Not Eligible  

## How SVM Works
1. Represent student data as feature vectors.  
2. Use SVM to find a **hyperplane** that best separates eligible vs non-eligible students.  
3. Predict eligibility for new students based on trained model.  

## Python Implementation (Simplified Example)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Example dataset (toy data)
data = {
    "income": [1000000, 3000000, 1500000, 5000000, 1200000, 4000000],
    "dependents": [4, 2, 5, 1, 3, 2],
    "attendance": [85, 95, 80, 98, 88, 97],
    "grade": [70, 85, 65, 90, 72, 88],
    "eligible": [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Features and label
X = df[["income", "dependents", "attendance", "grade"]]
y = df["eligible"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
