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

# messagesParsed.hist(column='len', by='label',bins=50)
# plt.show()

#DATA PREPROCESSING :
#SPLIT A FEATURE TEXT INTO ITS WORDS:
def split_into_tokens(feature):
    feature=str(feature)
    return TextBlob(feature).words

# print(messagesParsed.featureText.head().apply(split_into_tokens))

#FUNCTION TO NORMALIZE A WORD TO ITS BASE FORM
def split_to_lemma(feature):
    feature=str(feature).lower() #convert to lower because in spam filtering, uppercase or lowercase does not matter
    words=TextBlob(feature).words
    return[word.lemma for word in words]

# print(messagesParsed.featureText.head().apply(split_to_lemma))

#***********VECTORIZATION : MAGIC OF LINEAR ALGEBRA ********
# FIND THE TERM FREQUENCY
#BAG OF WORDS APPROACH
termFrequency = CountVectorizer(analyzer=split_to_lemma).fit(messagesParsed['featureText'])
# print("u" in termFrequency.vocabulary_)

termftest = termFrequency.transform([messagesParsed["featureText"][1]])
# print(messagesParsed["featureText"][1])
# print(termftest.shape)
# print(termFrequency.get_feature_names()[4421])

bag_words = termFrequency.transform(messagesParsed["featureText"])
tfidf_transformer = TfidfTransformer().fit(bag_words)
# sentence =  tfidf_transformer.transform(termftest)
# print(sentence)

#the word "the" will have a very low idf because it appears in almost every document
# print(tfidf_transformer.idf_[termFrequency.vocabulary_["the"]])

features_tfidf = tfidf_transformer.transform(bag_words)
print(features_tfidf) # prints (document_index,word) and tf-idf