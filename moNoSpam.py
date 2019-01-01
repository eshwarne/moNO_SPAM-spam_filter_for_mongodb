import matplotlib;
import matplotlib.pyplot as plt;
import csv;
from textblob import TextBlob;
#pandas to parse data
import pandas;
import sklearn;
import numpy as np;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC;
from sklearn.metrics import classification_report, f1_score, accuracy_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve;

#import corpus
message_text = [line.rstrip() for line in open('./dataset/spamData')]

messagesParsed = pandas.read_csv('./dataset/spamData', sep='\t', quoting=csv.QUOTE_NONE, names=["label","featureText"])

print(messagesParsed.head())

    
       