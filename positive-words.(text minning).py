# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:29:19 2022

@author: ashwi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
nltk.download('punkt')
nltk.download('stopwords') 
from nltk.corpus import stopwords

dataframe1=pd.read_csv("negative-words.txt",error_bad_lines=False,encoding="latin-1",header=None,)
df1=dataframe1.drop(dataframe1.index[0:26],axis=0)
df1


dataframe2=pd.read_csv("positive-words.txt",error_bad_lines=False,encoding="latin-1",header=None)
df2=dataframe2.drop(dataframe2.index[0:26],axis=0)
df2

# concating positive and negative word text

df=pd.concat([df1,df2],axis=0)

# Giving the name to column

df.columns={'X'}
df
##########################   sentimental analysis #################################

def calpolarity(x):
    return TextBlob(x).sentiment.polarity

def calSubjectivity(x):
    return TextBlob(x).sentiment.subjectivity

def segmentation(x):
    if x > 0:
        return "positive"
    if x== 0:
        return "neutral"
    else:
        return "negative"

df['polarity']=df["X"].apply(calpolarity)
df['subjectivity']=df["X"].apply(calSubjectivity)
df['segmentation']=df["polarity"].apply(segmentation)

df.head()

#Analysis and visualization

df.pivot_table(index=['segmentation'],aggfunc={"segmentation":'count'})

#The positive tweets are 440

#The negative tweets are 441

#The neutral tweets are 5908

#################################  Text Preprocessing  ###############################

# remove both the leading and the trailing characters

df = [X.strip() for X in df.X] 
df
# removes empty strings

book = [X for X in df if X] 
book[0:10]

# Joining the list into one string/text
text = ' '.join(book)
text

#Punctuation
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) 
no_punc_text

#Tokenization

from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])

# Removeing stopwords

my_stop_words = stopwords.words('english')

no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]

# Noramalize the data

lower_words = [X.lower() for X in no_stop_tokens]

print(lower_words[0:40])

#Stemming the data

from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]

print(stemmed_tokens[0:10])

import spacy

nlp = spacy.load('en_core_web_sm')
# lemmas being one of them, but mostly POS, which will follw later
doc = nlp(' '.join(no_stop_tokens))
print(doc[0:40])

lemmas = [token.lemma_ for token in doc]
print(doc[0:30])

################################  Feature Extraction  ######################################

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stemmed_tokens)

pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=False).head(20)
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names()[50:100])
print(X.toarray()[50:100])
print(X.toarray().shape)

# bigrams and trigrams
vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=(100))
bow_matix_ngram = vectorizer_ngram_range.fit_transform(book)

bow_matix_ngram 

print(vectorizer_ngram_range.get_feature_names())
print(bow_matix_ngram.toarray())
print(bow_matix_ngram.toarray().shape)

# TFidf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=(10))
tf_idf_matrix_n_gram_max_features = vectorizer_n_gram_max_features.fit_transform(book)
print(vectorizer_n_gram_max_features.get_feature_names())
print(tf_idf_matrix_n_gram_max_features.toarray())
print(tf_idf_matrix_n_gram_max_features.toarray().shape)

# wordcloud

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
def plot_cloud(wordcloud):
    plt.figure(figsize=(15,30))
    plt.imshow(wordcloud)
    plt.axis('off');

wordcloud = WordCloud(width=3000, height=2000,background_color="black",max_words=100,colormap='Set2').generate(text)

# plot
plot_cloud(wordcloud)






























































