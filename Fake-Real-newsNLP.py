#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


fake_news=pd.read_csv('Fake.csv')


# In[ ]:


real_news=pd.read_csv('True.csv')


# In[ ]:


fake_news.head()


# In[ ]:


fake_news.columns


# In[ ]:


real_news.head()


# In[ ]:


fake_news['class']='Fake'
real_news['class']='Real'


# In[ ]:


fake_news.head()


# In[ ]:


real_news.head()


# In[ ]:


fake_news['subject'].value_counts()


# In[ ]:


real_news['subject'].value_counts()


# In[ ]:


fake_news['date'].value_counts()


# In[ ]:


# we are going to deal with the 'title' and 'text' columns in this project,
# so we are going to drop the columns 'subject' and 'date'
fake_news.drop(['subject','date'],axis=1,inplace=True)
real_news.drop(['subject','date'],axis=1,inplace=True)


# In[ ]:


news=pd.concat([fake_news,real_news],ignore_index=True,sort=False)
news=news.sample(frac=1).reset_index(drop=True)
news


# In[ ]:


news['text']=news['title']+news['text']
news.drop('title',axis=1,inplace=True)
news.head()


# In[ ]:


news.describe()


# In[ ]:


news.groupby('class').describe().transpose()


# In[ ]:


news['length']=news['text'].apply(len)
news.head()


# In[ ]:


#EDA

news['length'].plot(bins=100, kind='hist',figsize=(14,7),)


# In[ ]:


news.length.describe()


# In[ ]:


news[news['length'] == 51892]['text'].iloc[0]


# In[ ]:


news[news['length'] == 51892]


# In[ ]:


news.hist(column='length', by='class', bins=70,figsize=(12,7))


# In[ ]:


#Text Pre-Processing
import string


# In[ ]:


import nltk


# In[ ]:


nltk.download_shell()


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


stopwords.words('english')[0:10]


# In[ ]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[ ]:


news.head()


# In[ ]:


news['text'].head(5).apply(text_process)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[ ]:


from sklearn.model_selection import train_test_split

nws_train, nws_test, class_train, class_test = train_test_split(news['text'], news['class'], test_size=0.3)

print(len(nws_train), len(nws_test), len(nws_train) + len(nws_test))
    
    


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[ ]:


pipeline.fit(nws_train,class_train)


# In[ ]:


predictionsNB = pipeline.predict(nws_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(predictionsNB,class_test))


# In[ ]:


print(confusion_matrix(predictionsNB,class_test))


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ SVM classifier
])


# In[ ]:


pipeline.fit(nws_train,class_train)


# In[ ]:


predictionsSVM = pipeline.predict(nws_test)


# In[ ]:


print(classification_report(predictionsSVM,class_test))


# In[ ]:


print(confusion_matrix(predictionsSVM,class_test))


# In[ ]:



param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', GridSearchCV(SVC(),param_grid,refit=True,verbose=3)),  # train on TF-IDF vectors w/ GridS classifier
])


# In[ ]:


pipeline.fit(nws_train,class_train)


# In[ ]:


grid_predictions = pipeline.predict(nws_test)


# In[ ]:


print(confusion_matrix(class_test,grid_predictions))


# In[ ]:


print(classification_report(class_test,grid_predictions))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




