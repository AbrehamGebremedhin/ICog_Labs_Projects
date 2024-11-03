import re
import nltk
import spacy
import string
import numpy as np
import pandas as pd
from pypdf import PdfReader
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

class Vectorizer:
    def __init__(self):
        english_stopset = list(set(stopwords.words('english')).union(
                  {"things", "that's", "something", "take", "don't", "may", "want", "you're",
                   "set", "might", "says", "including", "lot", "much", "said", "know",
                   "good", "step", "often", "going", "thing", "things", "think",
                   "back", "actually", "better", "look", "find", "right", "example",
                                                                  "verb", "verbs"}))
        self.lemmer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(analyzer='word',
                              ngram_range=(1, 2),
                              min_df=0.002,
                              max_df=0.99,
                              max_features=10000,
                              lowercase=True,
                              stop_words=english_stopset)
        self.kw_model = KeyBERT()
        
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip()
        return text

    def load_file(self, file_path):
        """
        Load and process a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            tuple: A tuple containing two lists:
                - docs (list of str): The lemmatized text of each page in the PDF.
                - keywords (list of list of str): The extracted keywords from each page in the PDF.
        """
        reader = PdfReader(file_path)
        docs = []
        keywords = []
        for page in range(len(reader.pages)):
            text = reader.pages[page].extract_text()
            cleaned_text = self.clean_text(text)
            lemmatized_text = ' '.join([self.lemmer.lemmatize(word) for word in cleaned_text.split()])
            docs.append(lemmatized_text)
            keywords.append(self.kw_model.extract_keywords(text, stop_words='english'))

        keywords = [[word for word, score in keyword_list] for keyword_list in keywords]

        return docs, keywords

    def get_similar_articles(self, query, top_result, file_path):
        """
        Find and return the most similar articles to a query from a PDF file.

        Args:
            query (str): The query string to search for.
            top_result (int): The number of top similar results to return.
            file_path (str): The path to the PDF file.

        Returns:
            tuple: A tuple containing three elements:
                - similarity_scores (list of float): The similarity scores of the top results.
                - docs (list of str): The lemmatized text of each page in the PDF.
                - keywords (list of list of str): The extracted keywords from each page in the PDF.
        """
        docs, keywords = self.load_file(file_path)
        X = self.vectorizer.fit_transform(docs)

        df = pd.DataFrame(X.T.toarray())

        tokenized_list = nltk.word_tokenize(query)

        q1 = ' '.join([self.lemmer.lemmatize(lemma_ops) for lemma_ops in tokenized_list])

        q = [q1]
        t = [q1]

        q_vec = self.vectorizer.transform(q).toarray().reshape(df.shape[0],)
        q_vect = self.vectorizer.transform(t).toarray().reshape(df.shape[0],)
        sim = {}
        titl = {}

        for i in range(len(docs)) and range(len(keywords)):                                            
            sim[i] = np.dot(df.loc[:, i].values, q_vec) / (np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec))  # Calculate the similarity
            titl[i] = np.dot(df.loc[:, i].values, q_vect) / (np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vect))

        sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)[:min(len(sim), top_result)]
        sim_sortedt = sorted(titl.items(), key=lambda x: x[1], reverse=True)[:min(len(titl), top_result)]

        similarity_scores = [score for _, score in sim_sorted]

        return similarity_scores, docs, keywords
        