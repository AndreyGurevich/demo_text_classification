from utils.helper import vanilla_tfidf, bigrams_tfidf, vanilla_flair, OptunaTFIDF
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from numpy import mean, std
from multiprocessing import freeze_support
import optuna

if __name__ == '__main__':
    freeze_support()
    le = LabelEncoder()

    df = pd.read_csv(Path(".") / "data" / "spam.csv",
                     names=["label", "text"],
                     header=0,
                     usecols=[0, 1],  # we need only two columns
                     encoding="ISO-8859-1",
                     on_bad_lines="warn")
    # print(df.head())
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

    # Train classical models with KFold validation
    for train_index, valid_index in skf.split(df["text"], df["target"]):
        # print("TRAIN:", train_index, "TEST:", valid_index)
        X_train = df.loc[train_index, "text"]
        X_valid = df.loc[valid_index, "text"]
        y_train = df.loc[train_index, "target"]
        y_valid = df.loc[valid_index, "target"]

        scores["vanilla_tfidf_with_sgd"].append(vanilla_tfidf(X_train, y_train, X_valid, y_valid))
        scores["bigrams_tfidf"].append(bigrams_tfidf(X_train, y_train, X_valid, y_valid))


    # Let's do 60/20/20 splitting
    # We can use eval_on_train_fraction option of trainer.train() method, but will do splitting manually to save
    # compatibility with possible additional frameworks (still looks like a little bit overengineering, but let it be).
    df = df.sample(frac=0.5)
    X_train, X_check, y_train, y_check = train_test_split(df["text"],
                                                          df["target"],
                                                          test_size=0.4,
                                                          random_state=42,
                                                          stratify=df["target"]
                                                          )
    X_valid, X_holdout, y_valid, y_holdout = train_test_split(X_check,
                                                              y_check,
                                                              test_size=0.5,
                                                              random_state=42,
                                                              stratify=y_check
                                                              )

    # vanilla_flair_score = vanilla_flair(X_train, y_train, X_valid, y_valid, X_holdout, y_holdout, 1)
    # scores["vanilla_flair"] = vanilla_flair_score

    study = optuna.create_study(direction="maximize")
    study.optimize(OptunaTFIDF(X_train, y_train, X_valid, y_valid), n_trials=100)
    print(study.best_trial)

    for key, value in scores.items():
        print(f"Method: {key}. Mean F1 score {mean(value):0.3f} with std {std(value):0.4f}")
