# demo_text_classification
This project compares different methods of text classification. It uses [SMS Spam Collection Data Set](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) downloaded from [Kaggle](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

## How to use
Assuming conda is your favorite environment manager
```
git clone git@github.com:AndreyGurevich/demo_text_classification.git
cd demo_text_classification
conda create --name demo_text_classification python==3.8
conda activate demo_text_classification
pip install -r requirements.txt
python text_classification.py
```
## Methodology
We can split data as 60%/20%/20% as train/validation/holdout. But dataset is rather small anf this type of splitting may lead to diverse subsets of text in splits. And this may lead to inaccurate scoring.

Another option is stratified K-Fold splitting. We will train several model and evaluate them. So we will get scores on all the data. If scores on different subset will have low deviation, than our trained model are quite robust on full dataset.

## Models
### vanilla_tfidf_with_sgd
TFIDF vectorization and Stochastic Gradient Descent with default parameters from scikit-learn package

No data preprocessing, no parameters tuning.

### preprocessed_tfidf_with_sgd
TFIDF vectorization and Stochastic Gradient Descent with default parameters from scikit-learn package

No data preprocessing, no parameters tuning.

## Results
