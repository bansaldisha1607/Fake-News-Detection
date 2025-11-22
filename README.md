#Fake News Detection
#Project Overview

with the growth of social media and online news websites, fake news has become an important problem.
It can spread misinformation, create panic, and influence public opionion in wrong way.

This project focuses on the problem and uses *Machine Learning* and **Natural Language Processor** to detect if a news article is false or not.

The model takes the *tesxt of a news article* as it's input and predicts its label.

---
#Objectives
collect and use a labelled fake/real news dataset.
Train a classification model (*Logistic Regression*)
Allow a user to type a news article and get a prediction whether it is fake or not.

#Project Structure
```text
fake-news-detection/
    data/
        news.csv
    src/
        train_model.py
model and saves it
        predict_single.py
for a single news text
    notebooks/
        fake_news_eda.ipynb
EDA + experiments
    README.md
    requirements.txt
Python libraries