# ML Project 1: Naive Bayes and Logistic Regression for Spam Classification

## Overview
This project implements spam email classification using:
- Logistic Regression (with L2 regularization)
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

## Requirements
- Python 3.9 or later

## Project Structure
```
ML-PROJECT-1/
├── dataset/                 # Raw email datasets (enron1, enron2, enron4)
├── dataset-train/           # Generated CSV files
│   ├── enron1/
│   │   ├── enron1_bow_train.csv
│   │   ├── enron1_bow_test.csv
│   │   ├── enron1_bernoulli_train.csv
│   │   └── enron1_bernoulli_test.csv
│   ├── enron2/
│   └── enron4/
├── models/
│   ├── log_regre.py         # Logistic Regression implementation
│   └── Naive_Bayes.py       # Naive Bayes implementations
├── feature.py               # Feature extraction functions
├── main.py                  # Data preprocessing and CSV generation
└── README.md
```

## Setup: Configuring Paths

Before running, update the base paths in each file: - easy way to do this is to right click on the dataset-train folder and click copy path option and use that for base path

### Windows
```python
base_path = "C:\\Users\\YourUsername\\path\\to\\ML-project-1\\dataset"
output_path = "C:\\Users\\YourUsername\\path\\to\\ML-project-1\\dataset-train"
```

### Mac/Linux
```python
base_path = "/Users/YourUsername/path/to/ML-project-1/dataset"
output_path = "/Users/YourUsername/path/to/ML-project-1/dataset-train"
```
## Python Environment Setup (optional but Recommended)

### Create virtual environment
```bash
python -m venv venv
```

### Activate virtual environment

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Install dependencies
```bash
pip install chardet numpy
```

### Select venv interpreter in VS Code
1. Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from `./venv/bin/python` (Mac) or `.\venv\Scripts\python.exe` (Windows)

## Running the Code

### Step 1: Generate Feature CSVs (Data Preprocessing) - don't think this is needed - as the featured csv are already there

This step converts raw email text files into numerical feature matrices (Bag of Words and Bernoulli representations).

```bash
python main.py
```

This generates 12 CSV files in the `dataset-train/` folder:
- `enron1_bow_train.csv`, `enron1_bow_test.csv`
- `enron1_bernoulli_train.csv`, `enron1_bernoulli_test.csv`
- (same for enron2 and enron4)

### Step 2: Run Logistic Regression

```bash
python models/log_regre.py
```

This trains Logistic Regression classifiers using:
- Three gradient descent variants: Batch GD, Mini-batch GD (batch size 50), SGD
- Both BoW and Bernoulli representations
- Hyperparameter tuning for λ (L2 regularization)

Output: Accuracy, Precision, Recall, F1 for all dataset/representation/GD variant combinations.

### Step 3: Run Naive Bayes

```bash
python models/Naive_Bayes.py
```

This trains:
- Multinomial Naive Bayes (on BoW data)
- Bernoulli Naive Bayes (on Bernoulli data)

Output: Accuracy, Precision, Recall, F1 for all datasets.

## Hyperparameters

### Logistic Regression
- **Lambda (λ) values tested:** 0.01, 0.1, 1.0, 10.0
- **Learning rate:** 0.1 (Batch GD), 0.01 (Mini-batch and SGD)
- **Epochs:** 500
- **Mini-batch size:** 50
- **Train/Validation split:** 70/30

Best λ is selected using validation accuracy, then model is retrained on full training data.

### Naive Bayes
- **Laplace smoothing (α):** 1 (add-one smoothing)
- All calculations performed in log-space to prevent numerical underflow.

## Output Format

Results are printed to console in the format:
```
dataset | representation | GD variant | λ=X | Acc=X.XXXX | Prec=X.XXXX | Rec=X.XXXX | F1=X.XXXX
```

## Notes
- Spam emails are labeled as 1 (positive class)
- Ham emails are labeled as 0
- Prediction threshold: P(spam) >= 0.5 → classify as spam