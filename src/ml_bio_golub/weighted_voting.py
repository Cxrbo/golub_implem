import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay


class weighted_voting_model:
    
    def weighted_voting(self, feature: str):
        """
        Implement the weighted voting for a feature in the dataset df: 0 = malignant, 1 = benign
        """
        
        # Cut the feature in two classs
        df_malignant = self.df.loc[self.df["diagnosis"] == "malignant", feature]
        df_benign = self.df.loc[self.df["diagnosis"] == "benign", feature]
        
        # Compute the mean and the standard deviation
        mean_mali = df_malignant.mean()
        mean_beni = df_benign.mean()
        
        st_dv_mali = df_malignant.std()
        st_dv_beni = df_benign.std()
        
        return (mean_mali - mean_beni) / (st_dv_beni + st_dv_mali), mean_mali, mean_beni
    
    def train(self, train_df):
        """
        Select the ten best features using the weighted_voting
        """
        self.df = train_df
        
        features = {}
        
        # Iterate on the number of feature
        for feature in self.df.columns:
            if feature == "sample_id" or feature == "diagnosis":
                continue
            
            w, mean_mali, mean_beni = self.weighted_voting(feature)
            
            features[feature] = (w, mean_mali, mean_beni)
        
        # Sort the features using numpy
        k = list(features.keys())
        v = np.array([value[0] for value in features.values()])

        idx = np.argsort(np.abs(v))[::-1][:10]
        features = {k[i]: features[k[i]] for i in idx}
        
        self.features = features
    
    def evaluate(self, test_df: pd.DataFrame):
        """
        Use to evaluate a dataframe Pandas and report the results using sklearn metrics
        """
        # The result predicted by the model
        pred = []
        scores = []
        # Iterate over the test dataframe
        for _, sample in test_df.iterrows():
            V = 0
            
            # Iterate over the feature selected in the training session
            for feature in self.features.items():
                # Compute the value for the sample
                name, value = feature
                w, mean_mali, mean_beni = value
                V += (w * (sample[name] - ((mean_beni + mean_mali) / 2)))

            scores.append(V)
            # Predict malignant or benign for the sample
            if V >= 0:
                pred.append("malignant")
            else:
                pred.append("benign")
        
        pred = np.array(pred)
        to_test = test_df["diagnosis"].to_numpy()
        
        to_test_num = np.where(to_test == "malignant", 1, 0)
        return classification_report(to_test, pred), balanced_accuracy_score(to_test, pred), (to_test_num, scores), confusion_matrix(to_test, pred, labels=["malignant", "benign"])
    
    