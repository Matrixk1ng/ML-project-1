import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def train_model(X, y, lr, epochs, lambda_param, batch_size=None):
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    
    if batch_size is None:
        batch_size = m
    
    for epoch in range(epochs):
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuf = X[indices]
        y_shuf = y[indices]
        
        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]
            batch_m = X_batch.shape[0]
            
            z = np.dot(X_batch, w) + b
            preds = sigmoid(z)
            error = preds - y_batch
            
            dw = (1/batch_m) * np.dot(X_batch.T, error) + (lambda_param/batch_m) * w
            db = (1/batch_m) * np.sum(error)
            
            w = w - lr * dw
            b = b - lr * db
    
    return w, b

def evaluate(X, y, w, b):
    probs = sigmoid(np.dot(X, w) + b)
    preds = (probs > 0.5).astype(int)
    
    TP = np.sum((preds == 1) & (y == 1))
    TN = np.sum((preds == 0) & (y == 0))
    FP = np.sum((preds == 1) & (y == 0))
    FN = np.sum((preds == 0) & (y == 1))
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return acc, prec, rec, f1

# Config
base_path = r"/Users/obinna/Desktop/ML project 1/dataset-train"
datasets = ["enron1", "enron2", "enron4"]
representations = ["bow", "bernoulli"]
gd_variants = [
    ("Batch GD", None, 0.1),      # name, batch_size, learning_rate
    ("Mini-batch 50", 50, 0.01),
    ("Mini-batch 100", 100, 0.01),
    ("SGD", 1, 0.01)
]
lambdas = [0.01, 0.1, 1.0, 10.0]

# Store all results
results = []

for dataset in datasets:
    for rep in representations:
        # Load data
        train_path = base_path + "//" + dataset + "//" + dataset + "_" + rep + "_train.csv"
        test_path = base_path + "//" + dataset + "//" + dataset + "_" + rep + "_test.csv"
        
        train_data = np.loadtxt(train_path, delimiter=',', skiprows=1)
        test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)
        
        X = train_data[:, :-1]
        y = train_data[:, -1]
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        
        # Shuffle and split 70/30
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_shuf = X[indices]
        y_shuf = y[indices]
        
        split = int(0.7 * len(y))
        X_train, X_val = X_shuf[:split], X_shuf[split:]
        y_train, y_val = y_shuf[:split], y_shuf[split:]
        
        for gd_name, batch_size, lr in gd_variants:
            # Find best lambda
            best_lambda = 0
            best_val_acc = 0
            
            for lam in lambdas:
                w, b = train_model(X_train, y_train, lr, 500, lam, batch_size)
                val_acc, _, _, _ = evaluate(X_val, y_val, w, b)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_lambda = lam
            
            # Retrain on full training data
            w_final, b_final = train_model(X_shuf, y_shuf, lr, 500, best_lambda, batch_size)
            
            # Evaluate on test
            acc, prec, rec, f1 = evaluate(X_test, y_test, w_final, b_final)
            
            results.append({
                "dataset": dataset,
                "rep": rep,
                "gd": gd_name,
                "lambda": best_lambda,
                "acc": acc,
                "prec": prec,
                "rec": rec,
                "f1": f1
            })
            
            print(f"{dataset} | {rep} | {gd_name} | λ={best_lambda} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")