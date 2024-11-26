# risk_level_classifier.py
"""
Script to classify transactions into risk levels using the Big Black Money Dataset.

Steps:
1. Data Cleaning and Preprocessing
2. Risk Categorization
3. Machine Learning Model Training (Random Forest)
4. Evaluation and Visualization
5. Model and Encoder Saving
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
file_path = 'Big_Black_Money_Dataset.csv'  # Ensure this file is in the same directory
dataset = pd.read_csv(file_path)

# Preprocessing
# Drop unnecessary columns
columns_to_drop = ['Transaction ID', 'Date of Transaction', 'Person Involved', 'Financial Institution']
dataset = dataset.drop(columns=columns_to_drop, axis=1)

# Encode categorical columns
categorical_columns = ['Country', 'Transaction Type', 'Industry', 'Destination Country', 'Source of Money', 'Tax Haven Country']
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    dataset[col] = label_encoders[col].fit_transform(dataset[col])

# Define risk levels based on Money Laundering Risk Score
def categorize_risk(score):
    if score <= 3:
        return 'Low'
    elif 4 <= score <= 6:
        return 'Medium'
    else:
        return 'High'

dataset['Risk Level'] = dataset['Money Laundering Risk Score'].apply(categorize_risk)
dataset = dataset.drop('Money Laundering Risk Score', axis=1)

# Split the data
X = dataset.drop('Risk Level', axis=1)
y = dataset['Risk Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize numerical columns
numerical_columns = ['Amount (USD)', 'Shell Companies Involved']
scaler = StandardScaler()
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Feature Importance
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# Save the trained model and encoders
joblib.dump(clf, 'risk_classifier_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
for col, encoder in label_encoders.items():
    joblib.dump(encoder, f'{col}_encoder.pkl')

print("Model and encoders saved!")
