from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from pathlib import Path
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import ClassificationCorpus

tempdir = Path(".") / "temp"
word_embeddings = [WordEmbeddings("glove"), FlairEmbeddings("news-forward-fast"), FlairEmbeddings("news-backward-fast")]
document_embeddings = DocumentLSTMEmbeddings(word_embeddings,
                                             hidden_size=512,
                                             reproject_words=True,
                                             reproject_words_dimension=256)


def vanilla_tfidf(x_train, x_valid, y_train, y_valid):
    vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                 max_df=1.0,
                                 min_df=1,
                                 )  # this is default parameters, just want them to be visible
    x_train_vectored = vectorizer.fit_transform(x_train)  # Fit on train data only
    x_valid_vectored = vectorizer.transform(x_valid)
    print(x_train_vectored.shape)
    sgd = SGDClassifier(random_state=42)
    sgd.fit(x_train_vectored, y_train)
    score = f1_score(y_true=y_valid, y_pred=sgd.predict(x_valid_vectored))
    return score


def bigrams_tfidf(x_train, x_valid, y_train, y_valid):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                 max_df=0.90,
                                 min_df=4,
                                 # max_features=5000
                                 )  # this is default parameters, just want them to be visible
    x_train_vectored = vectorizer.fit_transform(x_train)  # Fit on train data only
    x_valid_vectored = vectorizer.transform(x_valid)
    print(x_train_vectored.shape)
    sgd = SGDClassifier(random_state=42)
    sgd.fit(x_train_vectored, y_train)
    score = f1_score(y_true=y_valid, y_pred=sgd.predict(x_valid_vectored))
    return score


def prepare_temp_files_for_flair(x_train, x_valid, y_train, y_valid):
    # We should write dataset into files with format:
    # __label__<class_1> <text>
    # __label__<class_2> <text>
    tempdir.mkdir(exist_ok=True)
    with open(tempdir / "train.txt", "w", newline="\n", encoding="utf-8") as f:
        for text, label in zip(list(x_train), list(y_train)):
            f.write(f"__label__{label} {text}\n")

    with open(tempdir / "valid.txt", "w", newline="\n", encoding="utf-8") as f:
        for text, label in zip(list(x_valid), list(y_valid)):
            f.write(f"__label__{label} {text}\n")
    return None


def vanilla_flair(x_train, x_valid, y_train, y_valid):
    prepare_temp_files_for_flair(x_train, x_valid, y_train, y_valid)
    corpus = ClassificationCorpus(tempdir,
                                  test_file="valid.txt",
                                  # dev_file="dev.txt",
                                  train_file="train.txt",
                                  label_type="topic",
                                  )
    classifier = TextClassifier(document_embeddings,
                                label_dictionary=corpus.make_label_dictionary(label_type="topic"),
                                multi_label=False,
                                label_type="topic")
    trainer = ModelTrainer(classifier, corpus)
    trainer.train(tempdir, max_epochs=3)
