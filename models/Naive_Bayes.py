import numpy as np
base_path = "right click dataset-train folder and click copy path"
datasets = ["enron1", "enron2", "enron4"]
representations = ["bow", "bernoulli"]

def train_model_BoW(train_path, test_path, dataset):
    data_train = np.loadtxt(train_path, delimiter=',', skiprows=1)
    X = data_train[:, :-1]
    y = data_train[:, -1]

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

    vocab_size = X.shape[1] 

    total_spam_words = np.sum(word_counts_spam)
    total_ham_words = np.sum(word_counts_ham)

    laplace_smoothing = 1

    spam_probs_multinomial = (word_counts_spam + laplace_smoothing) / (total_spam_words + laplace_smoothing * vocab_size)
    ham_probs_multinomial = (word_counts_ham + laplace_smoothing) / (total_ham_words + laplace_smoothing * vocab_size)

    # prediction step for multinomial
    # Load test data
    test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)
    
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
    print(f"Multinomial Naive Bayes Evaluation: {dataset}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


def train_model_ber(train_path, test_path, dataset):
    #Bernoulli Naive Bayes
    train_data = np.loadtxt(train_path, delimiter=',', skiprows=1)
    X_ber = train_data[:, :-1]
    y_ber = train_data[:, -1]
    
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
    test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # Convert to binary
    X_test_binary = (X_test > 0).astype(int)

    predictions = []

    # Clip probabilities to avoid log(0) and log(>1)
    eps = 1e-10
    prob_spam_safe = np.clip(prob_word_given_spam, eps, 1 - eps)
    prob_ham_safe = np.clip(prob_word_given_ham, eps, 1 - eps)

    for i in range(len(X_test)):
        email = X_test_binary[i]
        
        log_spam = np.log(prob_spam_ber)
        log_ham = np.log(prob_ham_ber)
        
        log_spam = log_spam + np.sum(email * np.log(prob_spam_safe) + (1 - email) * np.log(1 - prob_spam_safe))
        log_ham = log_ham + np.sum(email * np.log(prob_ham_safe) + (1 - email) * np.log(1 - prob_ham_safe))
        
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
    print(f"Bernoulli Bayes Evaluation: {dataset}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


for dataset in datasets:
    for rep in representations:
        train_path = base_path + "\\" + dataset + "\\" + dataset + "_" + rep + "_train.csv"
        test_path = base_path + "\\" + dataset + "\\" + dataset + "_" + rep + "_test.csv"

        if rep == "bow":
            train_model_BoW(train_path, test_path, dataset)
        elif rep == "bernoulli":
            train_model_ber(train_path, test_path, dataset)