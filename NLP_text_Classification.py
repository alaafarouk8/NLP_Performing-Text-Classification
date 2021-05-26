# -*- coding: utf-8 -*-
"""
Created on Wed May 26 01:49:00 2021

@author: ALAA
"""
#import libraries
# container path  = C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_Performing-Text-Classification\dataset
import re
import nltk
import sklearn
from numpy import array
nltk.download('wordnet')
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = WordNetLemmatizer()
files = []


# Count Vectorizer
def getCountVectorizer_(f):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(f).toarray()
  #  print(vectorizer.get_feature_names())
    return X

# load files is done
def LoadFiles():
    dataset = sklearn.datasets.load_files(r"C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_Performing-Text-Classification\dataset", description=None, categories=None, load_content=True, shuffle=True, random_state=0)
    X , Y = dataset.data , dataset.target     # y is an array of 0s , 1s.
    return X , Y

# preparing data for generating tf-idf for the samples 
def  Preparedata(X):
    for i in range(len(X)):
         file = re.sub(r'\W', ' ', str(X[i]))
         file = re.sub(r'\s+[a-zA-Z]\s+', ' ', file)
         file = re.sub(r'\^[a-zA-Z]\s+', ' ', file) 
         file = re.sub(r'\s+', ' ', file, flags=re.I)
         file = re.sub(r'^b\s+', '', file)
         file = file.lower()
         file = file.split()
         file = [stemmer.lemmatize(word) for word in file]
         file = ' '.join(file)
         files.append(file)
    return files

#  generate tf-idf for the samples 
def generate_tfidf(files):
    tfidfconverter = TfidfVectorizer(max_features=1500,min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    x = tfidfconverter.fit_transform(files).toarray()
    return x

# divide the data into 20% test set and 80% training set.
def Splitingthedata(x,y):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, random_state=0)
    return Xtrain, Xtest, Ytrain, Ytest
    
x_ , y_ = LoadFiles() 
files = Preparedata(x_)
x = generate_tfidf(files)
x_train , x_test,y_train,y_test = Splitingthedata(x,y_)

classifier = LogisticRegression()
classifier.fit(x_train, y_train) 
yPredications = classifier.predict(x_test)
print("Accuracy: " , accuracy_score(y_test, yPredications))

"""
with open('text_classifier', 'wb') as picklefile:
    pickle.dump(classifier,picklefile)

with open('text_classifier', 'rb') as training_model:
    model = pickle.load(training_model)
 

w = input("enter word:")
zz = Preparedata(w)
y_pred2 = model.predict([zz])
print(y_pred2)

"""