# IBM-HR-Analytics-Employee-Attrition-Performance
In the business world, companies often face the challenge of retaining talented employees. One of the most pressing issues is the increasing rate of employee turnover, commonly known as HR attrition.
<br>
Author- Akanksha
<br>
import pandas as pd
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.dtypes)

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Attrition', data=df)
plt.show()
sns.histplot(df['Age'], bins=30, kde=True)
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(x='Department', hue='Attrition', data=df)
plt.show()
# plt.figure(figsize=(12, 8))
# sns.heatmap(df.corr(), annot=True, fmt='.2f')
# plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (adjust file name if needed)
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

#  Select only numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

#  Now calculate correlation only on numeric columns
corr_matrix = numeric_df.corr()

#  Plot the heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show()

# Building Predictive Models
# Data Splitting:
# Split the data into training and testing sets:

from sklearn.model_selection import train_test_split

X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
# Experiment with various models
# Logistic Regression

# Convert categorical columns to numeric using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

from sklearn.model_selection import train_test_split

# Target variable
y = df_encoded['Attrition_Yes']  # Since 'Attrition' became 'Attrition_Yes' after one-hot encoding

# Features
X = df_encoded.drop(['Attrition_Yes'], axis=1)

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Balance the dataset using class_weight='balanced'
# This tells the model to pay more attention to the minority class (those who left)

model = LogisticRegression(max_iter=1000, class_weight='balanced')

#Scale the features (especially important for logistic regression)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Random Forest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Evaluation
# Assess model performance using metrics like accuracy, precision, recall, and F1-score
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Model Interpretation
# Feature Importance:
# For tree-based models, extract feature importances:
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# SHAP Values:
# Use SHAP for detailed interpretation:
# import shap

# explainer = shap.Explainer(model, X_train)
# shap_values = explainer(X_test)
# shap.summary_plot(shap_values, X_test)

