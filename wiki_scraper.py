from bs4 import BeautifulSoup
from os import path
import pandas as pd
import urllib2 
#add def strings
class WikiScraper:

    def __init__(self,url):
        self.url = url
        self.page_links = None

    def _fetch_html(self):
        opener = urllib2.build_opener()
        resource = opener.open(self.url)
        data = resource.read()
        resource.close()
        soup = BeautifulSoup(data)
        return soup

class CategoryScraper(WikiScraper):
    
    def find_page_links(self):
        soup = self._fetch_html()
        #find the node in the DOM containing the page links
        page_node = soup.find('div',{"id":"mw-pages"})
        children = page_node.findChildren()
        page_links = []
        for child in children:
            if 'href' in child.attrs:
                page_links.append(child.get('href'))
        #don't count FAQ link and next page link
        self.page_links = [x for x in page_links if (":FAQ" not in x) and ("Category:" not in x)]
        #before return pages can check the last link pull those links if there's another page
        return self.page_links


class TextScraper(WikiScraper):
    
    def _clean_text(self,text):
        #can  also import re to remove tags '<>'
        return text.replace('\n',' ')
    
    def extract_text(self):
        soup = self._fetch_html()
        #extract relevant text but being careful not to extract the category section
        #This grabs the text in the <p> tags, but avoids the category text contained in a <div> tag
        text = " ".join([str(s.extract()) for s in soup('p')])
        text = self._clean_text(text)
        return text

class CategoryData:

    def __init__(self,categories,base_url,directory):
        #replace spaces with underscores in categories to fit wikipedia conventions
        self.categories = [c.replace(' ','_') for c in categories]
        self.base_url = base_url #https://en.wikipedia.org/wiki/Category:
        self.directory = directory
    def save_data_to_csv(self):
        data = []
        for cat in self.categories:
            cat_url = self.base_url + cat
            cat_scraper = CategoryScraper(cat_url)
            links = cat_scraper.find_page_links()
            for link in links:
                text_scraper = TextScraper('https://en.wikipedia.org'+link)
                text = text_scraper.extract_text()
                data.append((cat,text))
            print "added {} links of Data for category: {}".format(len(links),cat)
        df = pd.DataFrame.from_records(data=data, columns = ["category","text"])
        print "saving file"
        out_file = self.directory
        #use utf8 to avoid error letter when stemming
        df.to_csv(out_file,index=False,encoding = 'utf8')

if __name__ == "__main__":
    #local imports. Import config object
    from config import config
    categories = config.category_list
    path_dict=config.path_dict
    full_data=path_dict["full_data"]

    base_url ="https://en.wikipedia.org/wiki/Category:"
    data_object= CategoryData(categories=categories,base_url=base_url,directory=full_data)
    #print data_object.categories
    data_object.save_data_to_csv()
