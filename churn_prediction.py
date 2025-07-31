import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import os

# Create a directory for saving plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Display basic information
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert TotalCharges to numeric, handle empty strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print("\nTotalCharges after preprocessing:")
print(df['TotalCharges'].describe())

# Visualize churn distribution
print("\nGenerating Churn Distribution plot...")
sns.countplot(x='Churn', data=df, palette='coolwarm')
plt.title('Churn Distribution')
plt.xlabel('Churn (No = Stayed, Yes = Churned)')
plt.ylabel('Count')
plt.savefig('plots/churn_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("Churn Distribution plot saved.")

# Explore tenure distribution by churn
print("\nGenerating Tenure Distribution by Churn plot...")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
plt.title('Tenure Distribution by Churn')
plt.xlabel('Tenure (Months)')
plt.ylabel('Count')
plt.savefig('plots/tenure_distribution_by_churn.png', dpi=300, bbox_inches='tight')
plt.show()
print("Tenure Distribution by Churn plot saved.")

# Drop irrelevant columns
df.drop(['customerID'], axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Feature Engineering
df['AvgMonthlyChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
df['ContractTenure'] = df['Contract'] * df['tenure']

# Define features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyChargesPerTenure', 'ContractTenure']
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Address class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Verify balanced classes
print("Churn distribution after SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_balanced, y_train_balanced)

# Predict and evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
print("\nGenerating Feature Importance plot...")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Random Forest Model')
plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Feature Importance plot saved.")

# Export predictions for Power BI
test_results = X_test.copy()
test_results['ActualChurn'] = y_test
test_results['PredictedChurn'] = y_pred
test_results.to_csv('churn_predictions.csv', index=False)

# Export feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("\nExported predictions and feature importance to CSV files.")

# Simulate retention strategy
high_risk = test_results[test_results['PredictedChurn'] == 1]
print(f"\nHigh-risk customers (predicted to churn): {len(high_risk)}")
retained = int(0.15 * len(high_risk))
print(f"Potential retained customers: {retained}")
print(f"Churn reduction: {retained / len(test_results) * 100:.2f}%")