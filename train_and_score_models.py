import cPickle as pickle
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
###models to import
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


####helper functions
#need to fix encoding
# def stem_text(text):
#     '''
#     Takes a string text and returns text with all the words tokenized
#     '''
#     stemmer = SnowballStemmer("english")
#     #text = text.encode('utf-8')
#     word_list = text.split(" ")
#     return " ".join([stemmer.stem(w) for w in word_list])


def preprocess_data_frame(df):
    #drop rows that correspond with articles with two labels. Take last since first category has highest count
    old_len = df.shape[0]
    duplicates = df.duplicated(subset="text",take_last=True)
    df = df[duplicates == False]
    new_len = df.shape[0]
    print "dropped {} duplicates".format(old_len-new_len)
    df = df.reset_index()
    #apply any text transformation- trimming, article, stemming words, etc...
    df['text'] = df['text'].apply(lambda x: text_transforms(text=x))

    return df

def text_transforms(text,word_limit=500):
    #text = stem_text(text)[:word_limit]
    text = text[:word_limit]
    return text



def initialize_tfidf():
    tfidf = TfidfVectorizer(
        analyzer = u'word',
        ngram_range=(1,2),
        lowercase='true',
        stop_words = 'english',
        strip_accents = 'ascii',
        use_idf = True
    )
    return tfidf

def vectorize_data(train,test):
    '''
    Given the feature arrays train and test returns
    the tfidf vectorized arrays: X_train, X_test
    '''
    
    #perform tfidf vectorization
    tfidf = initialize_tfidf()
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


def initialize_models():
    '''
    Returns list of new models initialized 
    '''
    bayes_clf = MultinomialNB()
    random_forest_clf=RandomForestClassifier(
        n_estimators = 100,
        #max_depth = 5,
        min_samples_leaf =2
        )

    models = [bayes_clf,random_forest_clf]

    return models

def save_models(model_list,model_file="models.p"):
    pickle.dump(model_list, open(model_file, "wb" ) )


def compute_metrics():
    return

if __name__ == "__main__":
    #load data
    df = pd.read_csv('full_data.csv',encoding = 'utf8')
    df = preprocess_data_frame(df)
    X = df['text']
    y = df['category']
    models = initialize_models()
    #compute accuracy with cross validation
    for model in models:
        preds, pred_probs, scores=score_kfold_cv(
            model=model,
            X=X,
            y=y,
            num_folds=10)
        print scores
        print np.mean(scores)
    
    #Fit models to full data and save
    #initialize tfidf_vectorizer, fit it to the full data and save
    tfidf = initialize_tfidf()
    X=tfidf.fit_transform(X).toarray()
    pickle.dump(tfidf, open('tfidf.p', "wb" ) )
    #train models on full data
    models = initialize_models()
    for model in models:
        model.fit(X,y)
    #pick model list
    save_models(model_list=models,model_file="models.p")






