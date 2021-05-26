# -*- coding: utf-8 -*-
"""
Created on Wed May 26 01:49:00 2021

@author: ALAA
"""
#import libraries
import numpy as np
import re
import nltk
import sklearn
nltk.download('wordnet')
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
stemmer = WordNetLemmatizer()
files = []
# container path  = C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_Performing-Text-Classification\dataset

# load files is done
def LoadFiles():
    dataset = sklearn.datasets.load_files(r"C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_Performing-Text-Classification\dataset", description=None, categories=None, load_content=True, shuffle=True, random_state=0)
    X , Y = dataset.data , dataset.target     # y is an array of 0s , 1s.
    #print(X)
   # print ("print y" , Y)
    return X , Y
    

# preparing data for generating tf-idf for the samples 
def  Preparedata(X):
    for i in range(len(X)):
        file = re.sub(r'\W', ' ', str(X[i]))
        file = re.sub(r'\s+[a-zA-Z]\s+', ' ', file)
        file = re.sub(r'\s+', ' ', file, flags=re.I)
        file = re.sub(r'^b\s+', '', file)
        file = re.sub(r'^the\s+', '', file)
        file= file.lower()
        # lemmatization
        file=file.split()
        file= [stemmer.lemmatize(word) for word in file]
        file = ' '.join(file)
        print (file)
        files.append(file)
        return files
       
        






x , y = LoadFiles() 
Preparedata(x)    