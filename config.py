###models to import
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

#Class to configure parameters 
class Config:
    def __init__(self,
        category_list,
        model_dict,
        path_dict
        ):
        self.category_list = category_list
        self.model_dict = model_dict
        self.path_dict = path_dict

############
#models
bayes_clf = MultinomialNB()
random_forest_clf=RandomForestClassifier(
    n_estimators = 100,
    #max_depth = 5,
    min_samples_leaf =2
    )

category_list=["Rare diseases",
                "Infectious diseases",
                "Cancer",
                "Congenital disorders",
                "Organs (anatomy)",
                "Machine learning algorithms",
                "Medical devices"
                ]

model_dict = {
            "bayes_clf":bayes_clf,
            "random_forest_clf":random_forest_clf
            }

path_dict={
    "model_file":"models.p",
    "full_data":'full_data.csv',
    "tfidf_vectorizer":"tfidf.p"
}

config = Config(
    category_list=category_list,
    model_dict=model_dict,
    path_dict = path_dict
    )