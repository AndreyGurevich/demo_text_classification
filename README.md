# demo_text_classification
This project compares different methods of text classification. It uses [SMS Spam Collection Data Set](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) downloaded from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

## How to use
Assuming conda is your favorite environment manager
```
git clone git@github.com:AndreyGurevich/demo_text_classification.git
cd demo_text_classification
conda create --name demo_text_classification python==3.8
conda activate demo_text_classification
```
If you want to use GPU, install Pytorch with GPU support first
```
pip install -r requirements.txt
python text_classification.py
```
## Methodology
We can split data as 60%/20%/20% as train/validation/holdout. But dataset is rather small anf this type of splitting may lead to diverse subsets of text in splits. And this may lead to inaccurate scoring.

Another option is stratified K-Fold splitting. We will train several model and evaluate them. So we will get scores on all the data. If scores on different subset will have low deviation, than our trained model are quite robust on full dataset.

But we will use 60%/20%/20% approach with DL model just because this is faster to run and this is a demo task.

## Results
|Model|Mean F1|Std F1|
|---|---|---|
|TFIDF with SGD, all parameter by default|0.946|0.0056|
|TFIDF with SGD, tuned by intuition|0.951|0.0026|
|Flair with default parameter|0.875|NA|
|Flair, tuned by optuna|0.992|NA|