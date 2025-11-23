# Spam Classifier README

## Overview

This project demonstrates how to build a simple SMS spam classifier using Python, scikit-learn, and a Naive Bayes model. It includes data preparation, text vectorization, model training, evaluation, and saving/loading the trained model for future predictions.

## Features

* Prepares a small example dataset of SMS messages labeled as **spam** or **ham**.
* Splits data into training and testing sets.
* Converts text into numerical features using **CountVectorizer**.
* Trains a **Multinomial Naive Bayes** classifier.
* Evaluates model performance (accuracy + classification report).
* Saves the trained model and vectorizer using **joblib**.
* Loads saved model to classify new incoming messages.

## Requirements

* Python 3.x
* pandas
* scikit-learn
* joblib
* os (standard library)

Install dependencies:

```bash
pip install pandas scikit-learn joblib
```

## How to Use

1. Run the script to train the model.
2. The script will save two files:

   * `spam_model.pkl`
   * `vectorizer.pkl`
3. The script will then evaluate the model and classify a sample new message.

## File Descriptions

* **spam_model.pkl** – Trained Naive Bayes classifier.
* **vectorizer.pkl** – CountVectorizer fitted on the training data.

## Example Output

* Accuracy score on test data.
* Classification report (precision, recall, f1-score).
* Classification result for the sample new message.

## Notes

* This is a minimal example dataset. For real applications, use a larger labeled dataset.
* Naive Bayes works well for text classification but consider experimenting with TF-IDF and other algorithms for improved results.
