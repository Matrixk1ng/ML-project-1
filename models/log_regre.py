import numpy as np
# Step 1: load data from csv
data = np.loadtxt(r"C:\Users\Obinna\Documents\GitHub\ML-project-1\dataset-train\enron1\enron1_bow_train.csv", delimiter=',', skiprows=1)

X = data[:, :-1]
y = data[:, -1]

#Step 2: initilize weights
n_features = X.shape[1]

weights = np.zeros(n_features)
bias = 0.0
print(f"X shape: {X.shape}") # (rows, num_words)
print(f"y shape: {y.shape}") # (rows,)

#print(f"Weights shape: {weights.shape}") # Should match the number of words
#print(f"Bias value: {bias}")

#Step 3 sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1/ (1 +np.exp(-z))

# step 4 prediuction fuction
# TODO turn this into fucntion
linear_model = np.dot(X, weights) + bias
y_predicted = sigmoid(linear_model)
#print(y_predicted)


#step 5 tranining loop(gradient descent)
def train_model(X, y, lr, epochs, lambda_param):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for i in range(epochs):
        z = np.dot(X, w) + b
        predictions = sigmoid(z)
        error = predictions - y
        dw = (1/m) * np.dot(X.T, error) + (lambda_param/m) * w
        db = (1/m) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        if i % 100 == 0:
            # Simple cost calculation to see if we are improving
            # (This is just for us to see; the model doesn't use it directly)
            epsilon = 1e-15
            current_cost = -np.mean(y * np.log(predictions + epsilon) + (1-y) * np.log(1-predictions + epsilon))
            print(f"Epoch {i}: Cost {current_cost:.4f}")
    return w, b



#Step 7: Split training data
# 1. Shuffle data
# We shuffle an array of indices [0, 1, 2, ... len(y)-1]
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

# Reorder X and y using these shuffled indices
X_shuffled = X[indices]
y_shuffled = y[indices]

# 2. Split 70/30
split_idx = int(0.7 * len(y))

X_train, X_val = X_shuffled[:split_idx], X_shuffled[split_idx:]
y_train, y_val = y_shuffled[:split_idx], y_shuffled[split_idx:]

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")


lambdas = [0, 0.01, 0.1, 1, 10]
best_acc = 0
best_lambda = 0
best_w = None
best_b = None

for l in lambdas:
    # Train
    w, b = train_model(X_train, y_train, lr=0.1, epochs=500, lambda_param=l)
    
    # Validate
    val_probs = sigmoid(np.dot(X_val, w) + b)
    val_preds = (val_probs > 0.5).astype(int)
    
    # Simple Accuracy for selection
    acc = np.mean(val_preds == y_val)
    print(f"Lambda {l}: Val Accuracy {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_lambda = l
        best_w = w
        best_b = b

print(f"Best Lambda: {best_lambda}")

# 1. Get Final Predictions on Validation/Test Data
final_probs = sigmoid(np.dot(X_val, best_w) + best_b)
final_preds = (final_probs > 0.5).astype(int)

# 2. Calculate TP, TN, FP, FN
# We can use boolean math here
TP = np.sum((final_preds == 1) & (y_val == 1))
TN = np.sum((final_preds == 0) & (y_val == 0))
FP = np.sum((final_preds == 1) & (y_val == 0))
FN = np.sum((final_preds == 0) & (y_val == 1))

# 3. Calculate Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP+FP) > 0 else 0
recall = TP / (TP + FN) if (TP+FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision+recall) > 0 else 0

# print("--- Final Evaluation ---")
# print(f"Accuracy:  {accuracy:.4f}")
# print(f"Precision: {precision:.4f} (How many predicted spams were actually spam?)")
# print(f"Recall:    {recall:.4f}    (How many actual spams did we catch?)")
# print(f"F1 Score:  {f1_score:.4f}")

# After finding best_lambda
w_final, b_final = train_model(X_shuffled, y_shuffled, lr=0.1, epochs=500, lambda_param=best_lambda)

# rerun with best lambda
test_data = np.loadtxt(r"C:\Users\Obinna\Documents\GitHub\ML-project-1\dataset-train\enron1\enron1_bernoulli_test.csv", delimiter=',', skiprows=1)
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

final_probs = sigmoid(np.dot(X_test, w_final) + b_final)
final_preds = (final_probs > 0.5).astype(int)
# then calculate metrics on y_test

TP = np.sum((final_preds == 1) & (y_test == 1))
TN = np.sum((final_preds == 0) & (y_test == 0))
FP = np.sum((final_preds == 1) & (y_test == 0))
FN = np.sum((final_preds == 0) & (y_test == 1))

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("--- Test Set Evaluation ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1_score:.4f}")