from collections import defaultdict
import cPickle as pickle
from nltk.stem import SnowballStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics


####helper functions
#Doesn't seem to improve accuracy.  Also need to fix encoding for predict_categories.py.  
# def stem_text(text,word_limit):
#     '''
#     Takes a string text and returns text with all the words tokenized
#     '''
#     stemmer = SnowballStemmer("english")
#     #text = text.encode('utf-8')
#     word_list = text.split(" ")[:word_limit]
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

def text_transforms(text,word_limit=3000):
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
    full_labels=[]

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

        full_labels.extend(y_test)
        full_preds.extend(preds)
        full_pred_probs.extend(pred_probs)
        full_scores.append(score)
        
    return full_labels, full_preds, full_pred_probs, full_scores

def save_models(model_list,model_file="models.p"):
    pickle.dump(model_list, open(model_file, "wb" ) )


def score_ensemble(categories,labels,prob_dict):
    '''
    inputs: labels-- a list of the correct predictions
    prob_dict- a dict with keys being model_names and values the array of the
    predicted probabilites for each of their predictions
    output:
    score-- the accuracy for the ensemble found by taking the arg max of the average probabilities
    '''
    probs=np.array([prob_dict[model] for model in prob_dict.keys()]).mean(axis=0)
    pred_indices = np.apply_along_axis(np.argmax,axis=1,arr=probs)
    preds = [categories[i] for i in pred_indices]
    score = metrics.accuracy_score(y_true=labels,y_pred =preds)
    return score, preds

if __name__ == "__main__":
    #local imports. Import config object
    from config import config
    path_dict=config.path_dict
    model_file=path_dict["model_file"]
    tfidf_file=path_dict["tfidf_vectorizer"]
    full_data=path_dict["full_data"]
    prediction_data=path_dict["prediction_data"]
    model_dict = config.model_dict

    #load data
    df = pd.read_csv(full_data,encoding = 'utf8')
    df = preprocess_data_frame(df)
    X = df['text']
    y = df['category']
    
    #compute accuracy with cross validation
    full_labels = []
    full_probs = defaultdict(list)
    first_mod = model_dict.keys()[0]
    print "Mean accuracy across CV folds for each model:"
    for name,model in model_dict.iteritems():
        labels,preds, pred_probs, scores=score_kfold_cv(
            model=model,
            X=X,
            y=y,
            num_folds=10)
        print "\t {0}: {1}".format(name,np.mean(scores))
        if name == first_mod:
            #extract one copy of the labels
            full_labels.extend(labels)
        full_probs[name].extend(pred_probs)
    #get class names after we've trained the models
    categories = model_dict[first_mod].classes_
    ensemble_accuracy,ensemble_preds = score_ensemble(
        categories=categories,
        labels=full_labels,
        prob_dict=full_probs
    )
    print "Average accuracy of the ensemble: {}".format(ensemble_accuracy)
    #save labels and predictions to csv
    prediction_frame=pd.DataFrame(data={"labels":full_labels,"predictions":ensemble_preds})
    prediction_frame.to_csv(prediction_data,index=False)
    #initialize tfidf_vectorizer, fit it to the full data and save
    tfidf = initialize_tfidf()
    X=tfidf.fit_transform(X).toarray()
    pickle.dump(tfidf, open(tfidf_file, "wb" ) )
    
    #Fit models to full data and save    model_dict = config.model_dict
    model_dict = config.model_dict
    for name,model in model_dict.iteritems():
        model.fit(X,y)
    

    #pickle model dict
    save_models(model_list=model_dict,model_file=model_file)






