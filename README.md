# Healthcare Diagnosis Prediction System

This project implements a machine learning system for predicting medical conditions based on patient data. The system uses a deep neural network with residual connections to classify medical conditions from various patient features.

## Project Structure

```
Health-Care-Diagnosis-Prediction/
├── preprocess.py          # Data preprocessing and feature engineering
├── train.py              # Model training implementation
├── evaluate.py           # Model evaluation and metrics
├── model.pth             # Trained model weights
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Data Preprocessing (`preprocess.py`)

The preprocessing pipeline handles:
- Loading and parsing date columns
- Computing Length of Stay
- Removing irrelevant columns
- Feature encoding (categorical and numerical)
- Data splitting (80% train, 20% test)
- Saving preprocessed data and preprocessing objects

### Features Used
- Age
- Gender
- Blood Type
- Admission Type
- Billing Amount
- Length of Stay
- Medication
- Test Results

## Model Architecture (`train.py`)

The model uses a deep neural network with residual connections:

### Key Components:
1. **Input Layer**
   - Linear layer (input_size → 256)
   - Batch Normalization
   - ReLU activation
   - Dropout (0.3)

2. **Residual Blocks**
   - Two linear layers with batch normalization
   - Skip connections
   - ReLU activation
   - Dropout (0.4)
   - Dimensions: 256 → 128 → 64

3. **Output Layer**
   - Linear layer (64 → 32)
   - Batch Normalization
   - ReLU activation
   - Dropout (0.3)
   - Final layer (32 → num_classes)

### Training Process:
- Optimizer: AdamW with weight decay
- Learning Rate: OneCycleLR scheduler
- Loss Function: CrossEntropyLoss with class weights
- Batch Size: 64
- Early Stopping: 20 epochs patience
- Gradient Clipping: max_norm=1.0

## Model Evaluation (`evaluate.py`)

The evaluation script provides:
- Test accuracy
- Confusion matrix
- Classification report with:
  - Precision
  - Recall
  - F1-score
  - Support

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Preprocess the data:
```bash
python preprocess.py
```

2. Train the model:
```bash
python train.py
```

3. Evaluate the model:
```bash
python evaluate.py
```

## Dependencies

- Python 3.13+
- PyTorch 2.2.0+
- scikit-learn 1.4.0+
- pandas 2.2.0+
- numpy 1.26.0+
- joblib 1.3.0+

## Model Improvements

The model includes several advanced techniques to improve performance:

1. **Residual Connections**
   - Helps with gradient flow
   - Enables training of deeper networks
   - Improves feature propagation

2. **Regularization**
   - Dropout layers
   - Batch normalization
   - Weight decay
   - Gradient clipping

3. **Learning Rate Scheduling**
   - OneCycleLR scheduler
   - Cosine annealing
   - Warm-up period

4. **Class Imbalance Handling**
   - Class weights in loss function
   - Balanced data sampling

## Future Improvements

1. **Model Architecture**
   - Experiment with different network depths
   - Try attention mechanisms
   - Implement ensemble methods

2. **Data Processing**
   - Add more feature engineering
   - Implement data augmentation
   - Add feature selection

3. **Training Process**
   - Implement k-fold cross-validation
   - Add model checkpointing
   - Implement hyperparameter tuning

4. **Evaluation**
   - Add more evaluation metrics
   - Implement model interpretability
   - Add confidence scores

## Contributing

Feel free to submit issues and enhancement requests! 