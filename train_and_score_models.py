import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
###models to import
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


####helper functions
def stem_text(text):
    '''
    Takes a string text and returns text with all the words tokenized
    '''
    stemmer = SnowballStemmer("english")
    word_list = text.split(" ")
    return " ".join([stemmer.stem(w) for w in word_list])


def preprocess_data(df,word_limit=500):
    #drop rows that correspond with articles with two labels
    old_len = df.shape[0]
    duplicates = df.duplicated(subset="text")
    df = df[duplicates == False]
    new_len = df.shape[0]
    print "dropped {} duplicates".format(old_len-new_len)
    df = df.reset_index()
    #stem words
    df['text'] = df['text'].apply(lambda x: stem_text(x)[:word_limit])

    return df


def vectorize_data(train,test):
    '''
    Given the feature arrays train and test returns
    the tfidf vectorized arrays: X_train, X_test
    '''
    
    #perform tfidf vectorization
    tfidf = TfidfVectorizer(
            analyzer = u'word',
            ngram_range=(1,2),
            lowercase='true',
            stop_words = 'english',
            strip_accents = 'ascii',
            use_idf = True
        )
    X_train = tfidf.fit_transform(train)
    X_test = tfidf.transform(test)
    return X_train.toarray(), X_test.toarray()

def score_kfold_cv(model,X,y,num_folds):
    '''
    Given a model and data trains the model and predicts using stratified k-fold
    cross validation
    '''
    full_preds=[]
    full_pred_probs=[]
    full_scores=[]

    cv_folds = StratifiedKFold(y=y,n_folds=num_folds,shuffle=True,random_state=1)
    #train on k-1 fold and test on each remaining fold
    for train_index, test_index in cv_folds:
        X_train,X_test = vectorize_data(train=X[train_index],test=X[test_index])
        y_train=y[train_index]
        y_test = y[test_index]
        model.fit(X_train,y_train)
        
        preds = model.predict(X_test)
        pred_probs = model.predict_proba(X_test)
        score = metrics.accuracy_score(y_true=y_test,y_pred =preds)
        #score = np.mean([preds == y_test])

        full_preds.extend(preds)
        full_pred_probs.extend(pred_probs)
        full_scores.append(score)

    return full_preds, full_pred_probs, full_scores


def save_model():
    return

def compute_metrics():
    return

if __name__ == "__main__":
    #load data
    df = pd.read_csv('full_data.csv',encoding = 'utf8')
    df = preprocess_data(df)
    X = df['text']
    y = df['category']
    
    #initialize models
    bayes_clf = MultinomialNB()
    random_forest_clf=RandomForestClassifier(
        n_estimators = 100,
        #max_depth = 5,
        min_samples_leaf =2
        )

    models = [bayes_clf,random_forest_clf]
    for model in models:
        preds, pred_probs, scores=score_kfold_cv(
            model=model,
            X=X,
            y=y,
            num_folds=10)
        print scores
        print np.mean(scores)




