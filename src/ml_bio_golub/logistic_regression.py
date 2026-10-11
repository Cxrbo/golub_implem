import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing

def test_model_performance_regression(seed: int, df: pd.DataFrame, n_splits, n_repeats):
    """
    Compute the avg metrics for the 5 * 3 tests of the model using RepeatedStratifiedKFold
    """
    # Use of RepeatedStratifiedKFold from sklearn with 5 folds repeated 3 times
    rskf = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=seed
        )
    
    
    reports = []
    features_set = []
    features_keep = {col: 0 for col in df.columns if col not in ("sample_id", "diagnosis")}
    mean_accu = 0 
    mean_auc = 0
    mean_matrix = np.zeros((2, 2))
    
    # Iterate over the different fold
    for _, (train_idx, test_idx) in enumerate(rskf.split(df, df["diagnosis"])):
        # Initalize the scaler to normalize data
        scaler = preprocessing.StandardScaler()
        
        # Build dataframe for training and testing
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        X_train = scaler.fit_transform(df_train.drop(columns=["sample_id", "diagnosis"]))
        Y_train = df_train["diagnosis"]
        
        X_test = scaler.transform(df_test.drop(columns=["sample_id", "diagnosis"]))
        Y_test = df_test["diagnosis"]
        # Initalize the logistic regression model
        model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=seed)
        model.fit(X_train, Y_train)
        
        # Adapte the Y_test to get auc_score
        Y_test_num = np.where(Y_test == "malignant", 1, 0)
        
        # Predict the X_test df
        pred = model.predict(X_test)
        
        # Predict probabilities of sample of being malignant
        scores = model.predict_proba(X_test)[:, list(model.classes_).index("malignant")]
        
        # Compute the different score
        mean_accu += balanced_accuracy_score(Y_test, pred)
        mean_auc += roc_auc_score(Y_test_num, scores)
        mean_matrix += confusion_matrix(Y_test, pred, labels=["malignant", "benign"])
        
        # Append the report
        reports.append(classification_report(Y_test, pred))

    
    # Compute the mean
    mean_accu /= (n_repeats * n_splits)
    mean_auc /= (n_repeats * n_splits)
    mean_matrix /= (n_repeats * n_splits)
    
    # Save the results of each folds in log file
    if not os.path.exists("results/logistic_regression"):
            os.mkdir("results/logistic_regression")
    
    with open('results/logistic_regression/reports_logistic_regression.txt', 'w') as f:
        i = 1
        for rep in reports:
            f.write(f"Fold {i}\n")
            f.write(f"{rep}\n")
            i += 1
        
        f.write(f"==========================================\n")
        f.write(f"Average balanced accuracy: {round(mean_accu, 2)}, average auc score: {mean_auc}")
        f.close()
    
    # Save the average of the confusion matrix in results
    mat = ConfusionMatrixDisplay(confusion_matrix=mean_matrix, display_labels=["malignant", "benign"])
    mat.plot()
    plt.savefig("results/logistic_regression/avg_confusion_matrix.png")
    plt.close()

