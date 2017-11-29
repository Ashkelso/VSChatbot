import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC, SVC
import pickle

def prepareData(data_url, testSize):
    data = pd.read_excel(data_url)
    data = data.loc[data['labels'].isin(['assault', 'sexual abuse'])]
    X = data['data']
    y = data['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    return X_train, X_test, y_train, y_test, X, y

def makeModel(classifier, X_train, y_train):

    pipeline = Pipeline([
        ('bow', CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', classifier),  # train on TF-IDF vectors with  classifier
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

def testPerformance (model, model_type, X_test, y_test):
    print("results with "+model_type+" classifier: \n")
    print("confusion matrix: \n", confusion_matrix(y_test, model.predict(X_test)))

    print("\n"+classification_report(y_test, model.predict(X_test)))


#prepare training and testing data
data_url = "C:/Users/Ashley/Dropbox/cs109/SML Chatbot Project/training data.xlsx"
X_train, X_test, y_train, y_test, X, y = prepareData(data_url, testSize=0.2)

#######################
# testing the different classifiers
########################

#random forrest
print("Random Forest")
model = makeModel(RandomForestClassifier(), X_train, y_train)
testPerformance(model, "random forest", X_test, y_test)
print("\ncross validation: ")
ranforest = Pipeline([
            ('bow', CountVectorizer()),  # strings to token integer counts
            ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
            ('classifier', RandomForestClassifier()),  # train on TF-IDF vectors with  classifier
            ])
r = cross_val_score(ranforest, X_train, y_train, cv=3, scoring='accuracy')
print("mean accuracy: "+str(np.mean(r))+"\nstandard error: "+str(np.std(r)))

#New Support Vector Machine
print("\nNew Support Vector Machine")
model_1 = makeModel(NuSVC(), X_train, y_train)
testPerformance(model_1, "Suport Vector Machine", X_test, y_test)
print("\ncross validation: ")
NuSupport = Pipeline([
            ('bow', CountVectorizer()),  # strings to token integer counts
            ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
            ('classifier', NuSVC()),  # train on TF-IDF vectors with  classifier
            ])
s = cross_val_score(NuSupport, X_train, y_train, cv=3, scoring='accuracy')
print("mean accuracy: "+str(np.mean(s))+"\nstandard error:"+str(np.std(s)))

#Support vector machine
print("\nSupport Vector Machine")
model_2 = makeModel(SVC(), X_train, y_train)
testPerformance(model_2, "Suport Vector Machine", X_test, y_test)
print("\ncross validation: ")
Support = Pipeline([
            ('bow', CountVectorizer()),  # strings to token integer counts
            ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
            ('classifier', SVC()),  # train on TF-IDF vectors with  classifier
            ])
s = cross_val_score(Support, X_train, y_train, cv=3, scoring='accuracy')
print("mean accuracy: "+str(np.mean(s))+"\nstandard error:"+str(np.std(s)))

#Multi Naive Bayes
print("\nMultinomial Naive Bayes")
model_3 = makeModel(MultinomialNB(), X_train, y_train)
testPerformance(model_3, "MultinomialNB", X_test, y_test)
print("\ncross validation: ")
Bayes = Pipeline([
            ('bow', CountVectorizer()),  # strings to token integer counts
            ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
            ('classifier', MultinomialNB()),  # train on TF-IDF vectors with  classifier
            ])
n = cross_val_score(Bayes, X_train, y_train, cv=3, scoring='accuracy')
print("mean accuracy: "+str(np.mean(n))+"\nstandard error: "+str(np.std(n)))

################################
# Making the models
##############################

# # Training on 100% of the data
# forest = makeModel(RandomForestClassifier(), X, y)
# NewSVC = makeModel(NuSVC(), X, y)
# MultiNB = makeModel(MultinomialNB(), X, y)

# # Saving the classifiers
# pickle.dump(forest, open("randomForest.p","wb"))
# print("\nrandom forest saved")
# pickle.dump(NewSVC, open("NewSVC.p","wb"))
# print("\nsupport vector saved")
# pickle.dump(MultiNB, open("MultiNB.p","wb"))
# print("\nnaive bayes saved")