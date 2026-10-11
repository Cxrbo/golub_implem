import pandas as pd
import argparse
import os

from ml_bio_golub import explore_data, evalution_of_model, logistic_regression

SEED = 9


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weighted voting (Golub) vs regression logistique sur WDBC"
    )
    parser.add_argument(
        "data",
        nargs="?",
        default="data/data/wdbc.csv",
        help="Chemin du CSV (defaut : data/data/wdbc.csv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    os.makedirs("results", exist_ok=True)

    # Explore the data
    df = explore_data.analyse_data(df)

    # Compute the weighted voting model and metrics
    evalution_of_model.test_model_performance(SEED, df, 5, 3)

    # Compute the logistic regression model and metrics
    logistic_regression.test_model_performance_regression(SEED, df, 5, 3)
    
    print(f"Your results are in the results/ folder !")


if __name__ == "__main__":
    main()