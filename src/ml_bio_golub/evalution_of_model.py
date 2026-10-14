import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from itertools import combinations

from ml_bio_golub import weighted_voting

def test_model_performance(seed: int, df: pd.DataFrame, n_splits, n_repeats):
    """
    Compute the avg metrics for the 5 * 3 tests of the model using RepeatedStratifiedKFold
    """
    # Save the results of each folds in log file
    if not os.path.exists("results/weighted_voting"):
            os.mkdir("results/weighted_voting")
    
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
    for i, (train_idx, test_idx) in enumerate(rskf.split(df, df["diagnosis"])):
        # Build dataframe for training and testing
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        # Create a new model
        model = weighted_voting.weighted_voting_model()
        model.train(df_train)

        # Evaluate the model
        report, accu, (test, scores), matrix = model.evaluate(df_test)
        auc = roc_auc_score(test, scores)
        
        # Save the feature as set
        features_set.append(set(model.features.keys()))
        
        # Save the feature in each fold
        for k, _ in model.features.items():
            features_keep[k] += 1
        
        # Compute the add for each metrics
        mean_accu += accu
        mean_auc += auc
        mean_matrix += np.array(matrix)
        
        # Add the report
        reports.append(report)
        
        if i == 0:
            # Save the results of each folds in log file
            if not os.path.exists("results/weighted_voting/f1"):
                    os.mkdir("results/weighted_voting/f1")
            
            # Save the confusion matrix for the first fold
            mat = ConfusionMatrixDisplay(confusion_matrix=np.array(matrix), display_labels=["malignant", "benign"])
            mat.plot()
            plt.savefig("results/weighted_voting/f1/confusion_matrix.png")
            plt.close()
            
            # Save the auc curve
            RocCurveDisplay.from_predictions(test, scores)
            plt.title("AUC curve for weighted voting (from prediction)")
            plt.savefig("results/weighted_voting/f1/auc.png")
            
    # Compute the mean
    mean_accu /= (n_repeats * n_splits)
    mean_auc /= (n_repeats * n_splits)
    mean_matrix /= (n_repeats * n_splits)
    
    # Compute the mean jaccard score
    jaccard_scores = [
        len(a & b) / len(a | b)
        for a, b in combinations(features_set, 2)
    ]

    mean_jaccard = np.mean(jaccard_scores)
    
    with open('results/weighted_voting/reports_weighted_voting.txt', 'w') as f:
        i = 1
        for rep in reports:
            f.write(f"Fold {i}\n")
            f.write(f"{rep}\n")
            i += 1
        
        f.write(f"==========================================\n")
        f.write(f"Average balanced accuracy: {round(mean_accu, 2)}, average auc score: {round(mean_auc, 2)}, mean jaccard score: {mean_jaccard}")
        f.close()
    
    # Save the average of the confusion matrix in results
    mat = ConfusionMatrixDisplay(confusion_matrix=mean_matrix, display_labels=["malignant", "benign"])
    mat.plot()
    plt.savefig("results/weighted_voting/avg_confusion_matrix.png")
    plt.close()
    
    # Save the bar plot for the frequency of each feature
    freq = {k: v / (n_repeats * n_splits) * 100 for k, v in features_keep.items()}
    plt.figure(figsize=(12, 8))
    plt.bar(freq.keys(), freq.values())
    plt.xticks(rotation=90)
    plt.ylabel("Frequency (%)")
    plt.tight_layout()
    plt.savefig("results/weighted_voting/feature_frequency.png")