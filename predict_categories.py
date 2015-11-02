import cPickle as pickle
#import argparse
from wiki_scraper import TextScraper
from train_and_score_models import text_transforms

#make more modular.  Modify to print proper output.  need to extract category list


#helper_functions
def raw_data_to_feature_vector(text,tfidf):
    text = text_transforms(text)
    #transform operates on arrays, so place sample point in an array
    x = tfidf.transform([text])
    return x.toarray()


if __name__ == "__main__":
    #local imports. Import config object
    from config import config
    #load models trained on full data and fit tfidf_vectorizer
    path_dict=config.path_dict
    tfidf_file=path_dict["tfidf_vectorizer"]
    model_file=path_dict["model_file"]
    model_dict = pickle.load(open(model_file,"rb"))
    tfidf = pickle.load(open(tfidf_file,"rb"))
    base_url = "https://en.wikipedia.org/wiki/"
    while True:
        link = raw_input(
            "Please enter a topic for a valid wikipedia link:\n" 
             +"(i.e. https://en.wikipedia.org/wiki/***) to predict the category.\n"
             + "Type 'quit' to exit.\n"
            )
        if link == "quit":
            break
        else:
            if "en.wikipedia.org" not in link:
                link = base_url + link.replace(' ','_')
            print link
            #scrape link to return text
            text_scraper = TextScraper(link)
            text = text_scraper.extract_text()
            x=raw_data_to_feature_vector(text=text,tfidf=tfidf)
            #consider zipping models as dict to get names as well
            for name,model in model_dict.iteritems():
                pred_probs= model.predict_proba(x)
                classes = model.classes_
                print "Predicted probabilities for {} model:".format(name)
                for pair in zip(classes,pred_probs[0]):
                    print "\t {0}: {1:.2f}".format(pair[0],pair[1])