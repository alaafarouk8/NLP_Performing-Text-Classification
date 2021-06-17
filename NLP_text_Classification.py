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
import numpy as np
import numpy
import matplotlib.pyplot as plt
from numpy import array
nltk.download('wordnet')
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics import plot_roc_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import  accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = WordNetLemmatizer()
files = []
#####################################################################################
#count function 
def cnt(y_pred2):
    cntNeg = 0 
    cntPos = 0
    for i in range(len(y_pred2)):
        if (y_pred2[i]==0):
            cntNeg=cntNeg+1 
        else:
            cntPos=cntPos+1
    return cntNeg , cntPos 
#####################################################################################
def tokenizeText(text):
    text = text.lower()
    text = re.sub('[^\sa-zA-Z0-9ء-ي]', '', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text) 
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'^b\s+', '', text)
    return text.split()
#####################################################################################

# load files is done
def LoadFiles():
    positive , negative = 0 , 0
    dataset = sklearn.datasets.load_files(r"C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_Performing-Text-Classification\dataset", description=None, categories=None, load_content=True, shuffle=False, random_state=0)
    X , Y = dataset.data , dataset.target     # y is an array of 0s , 1s.
    for i in range(len(dataset.target)):
        if (dataset.target[i]==0):
            negative = negative+1
         #   print (" dataset[ " , i ," ]" , " is negative")
        else:
            positive = positive+1
       #     print (" dataset[ " , i ," ]"," is positive")
    return X , Y ,negative,positive
######################################################################################
# preparing data for generating tf-idf for the samples 
def  Preparedata(X):
    for i in range(len(X)):
         file = re.sub(r'\W', ' ', str(X[i]))
         file = re.sub(r'\s+[a-zA-Z]\s+', ' ', file)
         file = re.sub(r'\^[a-zA-Z]\s+', ' ', file) 
         file = re.sub(r'\s+', ' ', file, flags=re.I)
         file = file.lower()
         file = file.split()
         file = [stemmer.lemmatize(word) for word in file]
         file = ' '.join(file)
       #  print(file)
         files.append(file)
    return files
######################################################################################
# Count Vectorizer
def getCountVectorizer_(f):
    vectorizer = CountVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(f).toarray()
    return X
######################################################################################
#  generate tf-idf for the samples 
def getTFIDF(files):
    tfidfconverter = TfidfTransformer()
    X= tfidfconverter.fit_transform(files).toarray()
    return X
######################################################################################
# divide the data into 20% test set and 80% training set.
def Splitingthedata(x,y):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, random_state=0)
    return Xtrain, Xtest, Ytrain, Ytest
######################################################################################
# calling the functions
x_ , y_ ,pC , nC= LoadFiles() 
files = Preparedata(x_)
e=getCountVectorizer_(files)
x = getTFIDF(e)
x_train , x_test,y_train,y_test = Splitingthedata(x,y_)
classifier = LogisticRegression()
#classifier =svm.SVC(C=1.0, kernel='rbf', degree=3, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
classifier.fit(x_train, y_train) 
yPredications = classifier.predict(x_test)
print("Accuracy: %" , accuracy_score(y_test, yPredications)*100)


#####################################################################################
"""
mymodel = numpy.poly1d(numpy.polyfit(x_test,y_test, 3))

myline = numpy.linspace(0, 6, 100)

plt.scatter(x_test,y_test)
plt.plot(myline, mymodel(myline))
plt.show()
#####################################################################################

#plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':200})
plt.scatter(x_test, y_test)
plt.colorbar()
plt.title('Text Classifications')
plt.xlabel('X - value')
plt.ylabel('Y - value')
plt.show()
"""
#####################################################################################