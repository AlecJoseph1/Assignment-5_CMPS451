import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def loadDataset(file_path):

    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]  # features
    y = data[:, -1]   # labels
    return X, y

def trainHMMs(X_train, y_train, n_components=10):
    models = {}
    for digit in np.unique(y_train):
        digit_data = X_train[y_train == digit]
        
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
        model.fit(digit_data)
        models[int(digit)] = model
    return models

def predict(models, X_test):
    preds = []
    for sample in X_test:
        scores = []
        for digit, model in models.items():
            try:
                score = model.score(sample.reshape(1, -1))
            except:
                score = -np.inf
            scores.append((score, digit))
        pred_digit = max(scores)[1]
        preds.append(pred_digit)
    return np.array(preds)

# Main code
if __name__ == "__main__":
    dataset_path = "pendigits.tra" 

    X, y = loadDataset(dataset_path)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train models
    models = trainHMMs(X_train, y_train)

    # Predict
    y_pred = predict(models, X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")