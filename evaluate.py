import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from train import MedicalDiagnosisModel  # Import the model class

# Load preprocessed test data
X_test = joblib.load('X_test_preprocessed.pkl')
y_test = joblib.load('y_test_encoded.pkl')

# Convert data to PyTorch tensors
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Initialize the model
model = MedicalDiagnosisModel(X_test.shape[1], len(np.unique(y_test)))

# Load the saved model state
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    
    # Calculate accuracy
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    accuracy = 100 * correct / total
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, predicted.numpy())
    print('\nConfusion Matrix:')
    print(cm)
    
    # Generate classification report
    print('\nClassification Report:')
    print(classification_report(y_test, predicted.numpy()))

print("Evaluation complete.")