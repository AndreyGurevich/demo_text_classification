from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from pathlib import Path
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import ClassificationCorpus
import optuna

tempdir = Path(".") / "temp"
word_embeddings = [WordEmbeddings("glove"), FlairEmbeddings("news-forward-fast"), FlairEmbeddings("news-backward-fast")]
document_embeddings = DocumentLSTMEmbeddings(word_embeddings,
                                             hidden_size=512,
                                             reproject_words=True,
                                             reproject_words_dimension=256)


def vanilla_tfidf(x_train, y_train, x_valid, y_valid):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                 max_df=1.0,
                                 min_df=1,
                                 )  # this is default parameters, just want them to be visible
    x_train_vectored = vectorizer.fit_transform(x_train)  # Fit on train data only
    x_valid_vectored = vectorizer.transform(x_valid)
    sgd = SGDClassifier(random_state=42)
    sgd.fit(x_train_vectored, y_train)
    score = f1_score(y_true=y_valid, y_pred=sgd.predict(x_valid_vectored))

    return score


def bigrams_tfidf(x_train, y_train, x_valid, y_valid):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 max_df=0.90,
                                 min_df=4,
                                 )
    x_train_vectored = vectorizer.fit_transform(x_train)  # Fit on train data only
    x_valid_vectored = vectorizer.transform(x_valid)
    sgd = SGDClassifier(random_state=42)
    sgd.fit(x_train_vectored, y_train)
    score = f1_score(y_true=y_valid, y_pred=sgd.predict(x_valid_vectored))
    return score


def prepare_temp_files_for_flair(x_train, y_train, x_valid, y_valid, x_holdout, y_holdout):
    # We should write dataset into files with format:
    # __label__<class_1> <text>
    # __label__<class_2> <text>
    tempdir.mkdir(exist_ok=True)
    with open(tempdir / "train.txt", "w", newline="\n", encoding="utf-8") as f:
        for text, label in zip(list(x_train), list(y_train)):
            f.write(f"__label__{label} {text}\n")

    with open(tempdir / "dev.txt", "w", newline="\n", encoding="utf-8") as f:
        for text, label in zip(list(x_valid), list(y_valid)):
            f.write(f"__label__{label} {text}\n")

    with open(tempdir / "test.txt", "w", newline="\n", encoding="utf-8") as f:
        for text, label in zip(list(x_holdout), list(y_holdout)):
            f.write(f"__label__{label} {text}\n")
    return None


def vanilla_flair(x_train, y_train, x_valid, y_valid, x_holdout, y_holdout, max_epochs):
    prepare_temp_files_for_flair(x_train, y_train, x_valid, y_valid, x_holdout, y_holdout)
    corpus = ClassificationCorpus(tempdir,
                                  test_file="test.txt",
                                  dev_file="dev.txt",
                                  train_file="train.txt",
                                  label_type="topic",
                                  )
    classifier = TextClassifier(document_embeddings,
                                label_dictionary=corpus.make_label_dictionary(label_type="topic"),
                                multi_label=False,
                                label_type="topic")
    trainer = ModelTrainer(classifier, corpus)
    score = trainer.train(tempdir,
                  max_epochs=max_epochs,
                  num_workers=4)
    return score["test_score"]


class OptunaTFIDF(object):
    def __init__(self, x_train, y_train, x_valid, y_valid):
        print("y_train", y_train.shape)
        self.x_train = x_train
        self.x_valid = x_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        bigram_lower_bound = trial.suggest_int("bigram_lower_bound", 1, 4)
        bigram_upper_bound = trial.suggest_int("bigram_upper_bound", bigram_lower_bound, 4)
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.2)

        vectorizer = TfidfVectorizer(ngram_range=(bigram_lower_bound, bigram_upper_bound),
                                     max_df=1.0,
                                     min_df=1,
                                     )  # this is default parameters, just want them to be visible
        x_train_vectored = vectorizer.fit_transform(self.x_train)  # Fit on train data only
        x_valid_vectored = vectorizer.transform(self.x_valid)
        print(x_train_vectored.shape)
        sgd = SGDClassifier(random_state=42,
                            l1_ratio=l1_ratio)
        sgd.fit(x_train_vectored, self.y_train)
        score = f1_score(y_true=self.y_valid, y_pred=sgd.predict(x_valid_vectored))

        return score
