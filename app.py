from flask import Flask, render_template, jsonify, request, url_for
from flask_bootstrap import Bootstrap
import numpy as np
import pandas as pd
import time

#NPL
from textblob import TextBlob, Word
import re  ## To use Regular expression
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords ## removing all the stop words
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
import random
import pickle, pickletools, gzip


#Initialize the app
app = Flask(__name__)
Bootstrap(app)


@app.route('/')   #Add a decorator
def index():
    return render_template('index.html')

@app.route('/')   #Add a decorator
def results():
    return render_template('index.html')


@app.route('/analyse',methods=['POST'])  
def analyse():
    start = time.time()
    if request.method == 'POST':
        reviewText = request.form['reviewText']
        #NLP analysis
        blob = TextBlob(reviewText)
        received_text2 = blob
       #First received_text is linked with the index file, and received_text2 with the analysis
        blob_sentiment,blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
        number_of_tokens = len(list(blob.words))
        #splitting and adding the stemmed words except stopwords
        corpus = list()
        stop_words = set(stopwords.words('english'))
        for i in range(0, len(corpus)):
            review = re.sub('[^a-zA-Z]', ' ', ['reviewText'][i])
            review = review.split()
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in stop_words]
            review = ' '.join(review)
            corpus.append(review)
    end = time.time()
    final_time = end-start  
    summary = corpus
    return render_template('index.html', received_text=received_text2, number_of_tokens=number_of_tokens,blob_sentiment=blob_sentiment, blob_subjectivity=blob_subjectivity,summary=summary, final_time= final_time)


#To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    #train_set = pd.read_csv('train_set.csv')
    ###Loading model
    filepath = 'model.pkl'
    with gzip.open(filepath, 'rb') as f:
        p = pickle.Unpickler(f)
        gridsearch_svc_pipe = p.load()

    if request.method=='POST':
        reviewText = request.form['reviewText'] ##requesting new review from the input field
        review = re.sub('[^a-zA-Z]', ' ', reviewText)
        #review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        stop_words = set(stopwords.words('english'))
        review = [ps.stem(word) for word in review if not word in stop_words]
        review = ' '.join(review)
        corpus = [review]
        pred = gridsearch_svc_pipe.predict(corpus)
    return render_template('index.html',prediction=pred)


if __name__=='__main__':
    app.run(debug=True)