from utils.helper import vanilla_tfidf, bigrams_tfidf, vanilla_flair
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from numpy import mean, std
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    le = LabelEncoder()

    df = pd.read_csv(Path(".") / "data" / "spam.csv",
                     names=["label", "text"],
                     header=0,
                     usecols=[0, 1], # we need only two columns
                     encoding="ISO-8859-1",
                     on_bad_lines="warn")
    print(df.head())
    print(df["label"].value_counts())  # Check class balance: 4825 vs 747. Not bad, but slightly unbalanced

    assert df.loc[0, "label"] == "ham", "Something wrong with the datasource"


    # Let's do target encoding
    df["target"] = le.fit_transform(df["label"])


    # Use KFold because dataset is small and Stratified because it's unbalanced
    skf = StratifiedKFold(n_splits=5)
    scores = {
        "vanilla_tfidf_with_sgd": [],
        "bigrams_tfidf": [],
        "vanilla_flair": [],
    }
    for train_index, valid_index in skf.split(df["text"], df["target"]):
        # print("TRAIN:", train_index, "TEST:", valid_index)
        X_train = df.loc[train_index, "text"]
        X_valid = df.loc[valid_index, "text"]
        y_train = df.loc[train_index, "target"]
        y_valid = df.loc[valid_index, "target"]

        scores["vanilla_tfidf_with_sgd"].append(vanilla_tfidf(X_train, X_valid, y_train, y_valid))
        scores["bigrams_tfidf"].append(bigrams_tfidf(X_train, X_valid, y_train, y_valid))
        scores["vanilla_flair"].append(vanilla_flair(X_train, X_valid, y_train, y_valid))

    for key, value in scores.items():
        print(f"Approach: {key}. Mean F1 score {mean(value):0.3f} with std {std(value):0.4f}")
