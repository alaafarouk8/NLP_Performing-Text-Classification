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
from sklearn.datasets import load_files
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords
# container path  = C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_Performing-Text-Classification\dataset

# load files is done
def LoadFiles():
    dataset = load_files(r"C:\Users\ALAA\OneDrive\Documents\GitHub\NLP_Performing-Text-Classification\dataset")
    X , Y = dataset.data , dataset.target
    print(X)
    print ("print y" , Y)
    
    
    
LoadFiles()     