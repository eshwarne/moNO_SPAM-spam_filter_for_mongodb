# moNO_SPAM-spam_filter_for_mongodb
Using the techniques of NLP and machine learning we classify the messages in a chat as spam or not spam
# Abstract
With much spam and forward messages in the flow, though it cannot be eradicated, it is important to reduce it as much as possible. Also, we have to make sure that no redundant messages are sent often
We use python's TextBlob NLP to achieve spam filtering in the mongodb nosql database

# Technique Used 
We use the dataset which contains the data for spam and not spam labelled text messages. The dataset is cleaned and BOW transformed (Bag of words model is used to extract features from the text to represent its linguistic properties) so that the machine learning algorithm can work with the data. ML algorithms cannot work with raw text because they are not fixed or on a higher level, messy.

# Machine Learning Algorithm used:
Naive Bayes (Multinomial)