import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def analyse_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report different metrics on the dataset like size number of feature etc..
    Save a plot in results/data/malignant_vs_benign.png
    Save the report in results/data/report_data.txt
    return the df
    """
    
    # Result in dict
    result = {}

    # Size of dataframe
    s1 = df.size
    df = df.dropna()
    s2 = df.size
    
    result["Size before drop"] = s1
    result["Size after drop"] = s2

    # To know how many malignant and benign lines
    malignant = (df["diagnosis"] == "malignant").sum()
    benign = (df["diagnosis"] == "benign").sum()
    
    result["malignant"] = int(malignant)
    result["benign"] = int(benign)
    
    # BarPlot the comparaison
    plt.bar(["malignant", "benign"], [malignant, benign], label=["Number of malignant", "Number of benign"], color=["red", "blue"])
    plt.title("Compare between the number of malignant line and benign line")
    plt.ylabel("Nombre")
    plt.xlabel("benign or malignant")
    
    if not os.path.exists("results/data"):
        os.mkdir("results/data")
    
    plt.savefig("results/data/malignant_vs_benign.png", bbox_inches='tight')
    
    # Count the number of features
    tempdf = df.drop(columns=["sample_id", "diagnosis"])
    
    features = len(tempdf.columns)
    
    result["number of feature"] = features
    
    # Write the result in a file
    with open('results/data/reports_data.txt', 'w') as f:
            for k, v in result.items():
                f.write(f"{k}: {v}\n")
            f.close()
    
    return df
    
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Only one argument is needed")
    
    analyse_data(sys.argv[1])
