import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
import numpy as np

# Load the dataset, parsing date columns
data = pd.read_csv('healthcare_dataset.csv', parse_dates=['Date of Admission', 'Discharge Date'])

# Compute Length of Stay in days
data['Length of Stay'] = (data['Discharge Date'] - data['Date of Admission']).dt.days

# Drop irrelevant columns that are unlikely to predict medical condition
irrelevant_cols = ['Name', 'Date of Admission', 'Discharge Date', 'Doctor', 'Hospital', 
                   'Insurance Provider', 'Room Number']
data = data.drop(columns=irrelevant_cols)

# Separate features (X) and target (y)
X = data.drop('Medical Condition', axis=1)
y = data['Medical Condition']

# Encode the target variable (Medical Condition) into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define numerical and categorical columns
numerical_cols = ['Age', 'Billing Amount', 'Length of Stay']
categorical_cols = ['Gender', 'Blood Type', 'Admission Type', 'Medication', 'Test Results']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Scale numerical features
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)  # Encode categorical features
    ])

# Fit the preprocessor on training data
preprocessor.fit(X_train)

# Transform both training and test data
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Save preprocessed data and preprocessing objects
joblib.dump(X_train_preprocessed, 'X_train_preprocessed.pkl')
joblib.dump(y_train, 'y_train_encoded.pkl')
joblib.dump(X_test_preprocessed, 'X_test_preprocessed.pkl')
joblib.dump(y_test, 'y_test_encoded.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Preprocessing complete. Data saved.")