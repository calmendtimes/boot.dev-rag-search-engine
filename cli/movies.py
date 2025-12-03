
import json
import string 
from nltk.stem import PorterStemmer

class Movies:
    
    def __init__(self):
        self.movies_json_file_path = "data/movies.json"
        self.stopwords_file_path = "data/stopwords.txt"
        self.movies = self.load_movies()
        self.stopwords = self.load_stopwords()

    def load_movies(self):
        with open(self.movies_json_file_path) as f: 
            return json.load(f)

    def load_stopwords(self): 
        with open(self.stopwords_file_path) as f: 
            return f.read().split()

    def search_movies(self, query):
        found = []
        stemmer = PorterStemmer()
        translation = str.maketrans({c:'' for c in string.punctuation})
        for m in self.movies['movies']:
            title = m['title']
            translated = title.lower().translate(translation)
            q = query.lower().split()

            for w in q:
                w = stemmer.stem(w)
                if w in self.stopwords:
                    continue
                if w in translated:
                    found.append(title)
                    break

        return found