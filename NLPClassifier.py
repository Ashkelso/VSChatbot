# NLP classification engine

import pandas as pd
import pickle
import string
import nltk
 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC, SVC
from nltk.corpus import stopwords
 
lemma = nltk.wordnet.WordNetLemmatizer()
sno = nltk.stem.SnowballStemmer('english')
 
def prepareData (data_url, testSize):
    data = pd.read_excel(data_url)
    data = data.loc[data['labels'].isin(['assault', 'sexual abuse'])]
    X = data['data']
    y = data['labels']
 
    # since data is small and labels are imbalanced
    # shuffle before splitting and use y label to stratify the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, stratify=y, shuffle=True)
    return X_train, X_test, y_train, y_test, X, y
 
def preprocessText (message):
    # removess any punctuation
    nopunc = [char for char in message if char not in string.punctuation]
 
    # forms a string without punctuation
    nopunc = ''.join(nopunc)
 
    # removes any stopwords and returns the rest as list of words
    nostop = [word for word in nopunc.split()  if word.lower() not in stopwords.words('english')]
    return nostop
 
def preprocessTextAdvanced (message):
    # removes punctation and stop words
    nopunc_stop = preprocessText(message)
 
    # lemmatising
    lemmatised = [lemma.lemmatize(word) for word in nopunc_stop]
 
    # stemming
    stemmed = [sno.stem(word) for word in lemmatised]
 
    # removes any stopwords again after stemming which may have exposed stopwords which were contracted
    stemmed_nostop = [word for word in stemmed if word.lower() not in stopwords.words('english')]
    return stemmed_nostop
 
def getPipeline (classifier):
    print('\nUsing pipeline without text preprocessing')
    pipeline = Pipeline([
        ('vec', CountVectorizer()),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', classifier),  # train on TF-IDF vectors with  classifier
    ])
    return pipeline
 
def getPipelineAdvanced (classifier):
    print('\nUsing pipeline with text preprocessing')
    pipeline = Pipeline([
        ('vec', CountVectorizer(analyzer = preprocessTextAdvanced)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', classifier),  # train on TF-IDF vectors with  classifier
    ])
    return pipeline
 
def trainModel (pipeline, name, X_train, y_train, params, numkfold):
    print('\nTraining with', name)
    best_model = GridSearchCV(pipeline, param_grid=params, cv=numkfold, n_jobs=-1, verbose=1, scoring='accuracy')
    best_model.fit(X_train, y_train)
    best_accuray = best_model.cv_results_['mean_test_score'][best_model.best_index_]
    best_std = best_model.cv_results_['std_test_score'][best_model.best_index_]
    print('best k-fold index:', best_model.best_index_,
          '\tmean accuracy:', str(best_accuray),
          '\tstd:', str(best_std))
    #if name == 'randomForest':
    #    print('oob score:', best_model.oob_score_)
    print('hyperparameters:', best_model.best_params_)
    print('best estimator:', best_model.best_estimator_)
    return best_model
 
def refitModel (pipeline, name, X, y, params, numkfold):
    print('\nRefitting', name)
    refitted_model = GridSearchCV(pipeline, param_grid=params, cv=numkfold, n_jobs=-1, verbose=0, scoring='accuracy')
    refitted_model.fit(X, y)
    refitted_accuray = refitted_model.cv_results_['mean_test_score'][refitted_model.best_index_]
    refitted_std = refitted_model.cv_results_['std_test_score'][refitted_model.best_index_]
    print('mean accuracy:', str(refitted_accuray), '\tstd:', str(refitted_std))
    return refitted_model
 
def testPerformance (model, name, X_test, y_test):
    print('\nTesting with', name)
    y_predicted = model.predict(X_test)
    #print('test label\tpredicted label')
    #for i in range(len(y_predicted)):
    #    print(y_test.values[i], '\t', y_predicted[i])
    test_accuracy = model.score(X_test, y_test)
    print('mean accuracy:', str(test_accuracy))
    print('confusion matrix:\n', confusion_matrix(y_test, y_predicted))
    print('classification report:\n', classification_report(y_test, y_predicted))
    print('..............................')
    return test_accuracy
 
def main():
    #prepare training and testing data
    data_url = 'training data.xlsx'
    X_train, X_test, y_train, y_test, X, y = prepareData(data_url, testSize=0.2)
 
    # define models and their parameters for hyperparameter grid search
    # dictionary key name is also used as a filename to save the model
    models = {
        'RandomForest': (
            RandomForestClassifier(),
            {
#                'classifier__max_depth': [10, 50, 100],
                'classifier__n_estimators': [100, 500, 1000],
               'classifier__oob_score': [True, False]
            }
        ),
        'NuSVC': (
            NuSVC(),
            {
                'classifier__gamma': [1e-4, 1e-3, 1e-2],
                'classifier__nu': [0.1, 0.3, 0.5, 0.7],
            }
        ),
        'SVC': (
            SVC(),
            {
                'classifier__C': [1, 10, 100],
                'classifier__gamma': [1e-2, 1e-1, 0],
            }
        ),
        'MultiNB': (
            MultinomialNB(),
            {
                'classifier__alpha': [1e-2, 1e-1, 1],
                'classifier__fit_prior': [True, False],
            }
        ),
    }
 
    # exersise all models
    numkfold = 10
    index_names = ['Without','With']
    df_train_results = pd.DataFrame(index=index_names)
    df_test_results = pd.DataFrame(index=index_names)

    for name, (classifier, params) in models.items():
        pipeline_params = {}
 
        # add parameters for pre-processing
        pipeline_params['vec__ngram_range'] = [(1, 1), (1, 2)]
 
        # Adding model params to the pipeline for hyperparameter grid search
        pipeline_params.update(params)
 
        # Training and testing the model without text preprocessing for comparison
        pipeline = getPipeline(classifier)
        hyper_model = trainModel(pipeline, name, X_train, y_train, pipeline_params, numkfold)
        test_accuracy = testPerformance(hyper_model, name, X_test, y_test) #testing with untouch data
        df_train_results.loc[index_names[0], name] = hyper_model.cv_results_['mean_test_score'][hyper_model.best_index_] * 100
        df_test_results.loc[index_names[0], name] = test_accuracy * 100

        # Training and testing the model again with text preprocessig for comparison
        pipeline = getPipelineAdvanced(classifier)
        hyper_model = trainModel(pipeline, name, X_train, y_train, pipeline_params, numkfold)
        test_accuracy = testPerformance(hyper_model, name, X_test, y_test) #testing with untouch data
        df_train_results.loc[index_names[1], name] = hyper_model.cv_results_['mean_test_score'][hyper_model.best_index_] * 100
        df_test_results.loc[index_names[1], name] = test_accuracy * 100

        # Wrapping hyperparameters returned from the last training as an array
        # so it can be used to refit the model through a grid search
        hyperparameter = {}
        for key, value in hyper_model.best_params_.items():
            hyperparameter[key] = [value]
 
        # Refitting with 100% of the data for production
        hyper_model = refitModel(pipeline, name, X, y, hyperparameter, numkfold)
 
        # Saving the model for each classifier
        filename = name + '.p'
        pickle.dump(hyper_model, open(filename,'wb'))
        print('\nSaved as', filename)
        print('\n==============================')

    print('\nTraining Summary (mean accuracy %):\n', df_train_results)
    print('\nTest Summary (mean accuracy %):\n', df_test_results)

if __name__ == '__main__':
    # main function will not be called when grid search jobs are run in parallel
    main()
