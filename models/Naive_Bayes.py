import numpy as np
# You don't need chardet or pathlib if you rely on the CSV structure
# from pathlib import Path 

# 1. FIX: Use loadtxt for CSV
dataset = np.loadtxt(r"/Users/obinna/Desktop/ML project 1/dataset-train/enron1/enron1_bow_train.csv", delimiter=',', skiprows=1)
dataset_bernoulli = np.loadtxt(r"/Users/obinna/Desktop/ML project 1/dataset-train/enron1/enron1_bernoulli_train.csv", delimiter=',', skiprows=1)


X = dataset[:, :-1]
y = dataset[:, -1]

# Count spam and ham emails
spam_count = np.sum(y == 1)
ham_count = np.sum(y == 0)
email_count = len(y)

# Calculate priors
prob_spam_multi = spam_count / email_count
prob_ham_multi = ham_count / email_count

# Step 4: Calculate word probabilities
# Summing columns gives the total count of each specific word
word_counts_spam = np.sum(X[y == 1], axis=0)
word_counts_ham = np.sum(X[y == 0], axis=0)

# 2. FIX: Get Vocab Size directly from X matrix
# The number of columns in X is your vocabulary size
vocab_size = X.shape[1] 

# Calculate total word volume (sum of the row sums)
total_spam_words = np.sum(word_counts_spam)
total_ham_words = np.sum(word_counts_ham)

# Laplace Smoothing
laplace_smoothing = 1

spam_probs_multinomial = (word_counts_spam + laplace_smoothing) / (total_spam_words + laplace_smoothing * vocab_size)
ham_probs_multinomial = (word_counts_ham + laplace_smoothing) / (total_ham_words + laplace_smoothing * vocab_size)
# prediction step for multinomial
# Load test data
test_data = np.loadtxt(r"/Users/obinna/Desktop/ML project 1/dataset-train/enron1/enron1_bow_test.csv", delimiter=',', skiprows=1)
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Predict for each test email
predictions = []

for i in range(len(X_test)):
    email = X_test[i]
    
    # Log probabilities (add logs instead of multiplying)
    log_spam = np.log(prob_spam_multi)
    log_ham = np.log(prob_ham_multi)
    
    # For each word, add count * log(probability)
    log_spam = log_spam + np.sum(email * np.log(spam_probs_multinomial + 1e-10))  # Add small value to avoid log(0)
    log_ham = log_ham + np.sum(email * np.log(ham_probs_multinomial + 1e-10))
    
    # Compare and decide
    if log_spam > log_ham:
        predictions.append(1)
    else:
        predictions.append(0)

predictions = np.array(predictions)

# Evaluate
TP = np.sum((predictions == 1) & (y_test == 1))
TN = np.sum((predictions == 0) & (y_test == 0))
FP = np.sum((predictions == 1) & (y_test == 0))
FN = np.sum((predictions == 0) & (y_test == 1))

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print("Multinomial Naive Bayes Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")



#Bernoulli Naive Bayes
X_ber = dataset_bernoulli[:, :-1]
y_ber = dataset_bernoulli[:, -1]

# Convert to binary (should already be 0/1, but just in case)
X_bernoulli = (X_ber > 0).astype(int)

# Count spam and ham emails
n_spam = np.sum(y_ber == 1)
n_ham = np.sum(y_ber == 0)

# Count how many spam/ham emails contain each word
word_presence_spam = np.sum(X_bernoulli[y_ber == 1], axis=0)
word_presence_ham = np.sum(X_bernoulli[y_ber == 0], axis=0)

# Laplace smoothing
alpha = 1
prob_word_given_spam = (word_presence_spam + alpha) / (n_spam + 2 * alpha)
prob_word_given_ham = (word_presence_ham + alpha) / (n_ham + 2 * alpha)

# Class priors
prob_spam_ber = n_spam / len(y_ber)
prob_ham_ber = n_ham / len(y_ber)

#step 5 predicting on test data
# Load test data
test_data = np.loadtxt(r"/Users/obinna/Desktop/ML project 1/dataset-train/enron1/enron1_bernoulli_test.csv", delimiter=',', skiprows=1)
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Convert to binary
X_test_binary = (X_test > 0).astype(int)

predictions = []

for i in range(len(X_test)):
    email = X_test_binary[i]
    
    log_spam = np.log(prob_spam_ber)
    log_ham = np.log(prob_ham_ber)
    
    # When word is present: add log(prob)
    # When word is absent: add log(1 - prob)
    log_spam = log_spam + np.sum(email * np.log(prob_word_given_spam + 1e-10) + (1 - email) * np.log(1 - prob_word_given_spam + 1e-10))
    log_ham = log_ham + np.sum(email * np.log(prob_word_given_ham + 1e-10) + (1 - email) * np.log(1 - prob_word_given_ham + 1e-10))
    
    if log_spam > log_ham:
        predictions.append(1)
    else:
        predictions.append(0)

predictions = np.array(predictions)

# Evaluate
TP = np.sum((predictions == 1) & (y_test == 1))
TN = np.sum((predictions == 0) & (y_test == 0))
FP = np.sum((predictions == 1) & (y_test == 0))
FN = np.sum((predictions == 0) & (y_test == 1))

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
print("Bernoulli Bayes Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")