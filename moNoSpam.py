from matplotlib.pyplot import plot
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
messagesParsed["len"] = messagesParsed['featureText'].map(lambda message : len(message))
# print(messagesParsed.head())
# fig1 = messagesParsed.len.plot(kind = 'hist')
# plt.show()

print(messagesParsed.len.describe())
print(list(messagesParsed.featureText[messagesParsed.len == 910]))

messagesParsed.hist(column='len', by='label',bins=50)
plt.show()

#DATA PREPROCESSING :
    
       