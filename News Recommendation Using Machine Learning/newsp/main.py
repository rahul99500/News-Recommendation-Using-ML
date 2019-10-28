from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug import secure_filename
from passlib.hash import sha256_crypt

app = Flask(__name__)

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
# from imblearn.over_sampling import SMOTE
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


from datetime import datetime, timedelta

import time
import os
import random
# from document import Document
from collections import defaultdict, Counter

# from wordcloud import WordCloud

from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key='6b4e60b44b5746c3a5f75b18fccba442')

t_path = "../dataset/bbc/"

all_docs = defaultdict(lambda: list())

topic_list = list()
text_list = list()

print("Reading all the documents...\n")

for topic in os.listdir(t_path):
    d_path = t_path + topic + '/'

    for f in os.listdir(d_path):
        f_path = d_path + f
        file = open(f_path,'r')
        text_list.append(file.read())
        file.close()
        topic_list.append(topic)



#splitting the data training and test data
title_train, title_test, category_train, category_test = train_test_split(text_list,topic_list)
title_train, title_dev, category_train, category_dev = train_test_split(title_train,category_train)

print("Training: ",len(title_train))
print("Developement: ",len(title_dev),)
print("Testing: ",len(title_test))

tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
stop_words = nltk.corpus.stopwords.words("english")
vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words)
vectorizer.fit(iter(title_train))
Xtr = vectorizer.transform(iter(title_train))
Xde = vectorizer.transform(iter(title_dev))
Xte = vectorizer.transform(iter(title_test))



encoder = LabelEncoder()
encoder.fit(category_train)
Ytr = encoder.transform(category_train)

Yde = encoder.transform(category_dev)
Yte = encoder.transform(category_test)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

from sklearn.svm import SVC
svm = SVC(C= 10000000.0, gamma='auto', kernel='rbf')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=17)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

import requests

classifier_list = [nb,svm,knn,lr,dt,rfc]
max_score = {}
classifier_names = ['naive bayes','support vector machine','k-nearest neighbour','logistic regression','decision tree','randomforestclassifier']
prediction_list = list();
for i in range(len(classifier_list)):
    print('\nClassifier name :',classifier_names[i])

    classifier = classifier_list[i]

    print('\nTraining...',classifier_names[i])
    classifier.fit(Xtr,Ytr)

    print('\nerror(confusion) matrix of',classifier_names[i])
    pred = classifier.predict(Xde)
    prediction_list.append(pred[0]);
    print('predicted index for category ', pred[0])
    print(classification_report(Yde,pred,
    target_names=encoder.classes_))
    score = classifier.score(Xte,Yte)
    print(classifier_names[i],'score is =>',score)
    print('====================================================\n')
    max_score[classifier_names[i]]=score

prediction_list = Counter(prediction_list)
print('prediction list',prediction_list.most_common())

keys = list(max_score.keys())
values = list(max_score.values())
print('max score classifier name :',keys[values.index(max(values))],' and score:',max(values))

print('\n\n================================================================\n\n')
print('\t\t\tRecommendation System\n\n')


url = ('https://newsapi.org/v2/top-headlines?'
   'country=in&'
   'apiKey=6b4e60b44b5746c3a5f75b18fccba442')
response = requests.get(url)
json= response.json()
# step 2
Trending_articles = defaultdict(dict)

for article in json['articles']:
#     print(article['source']['name'])
    Trending_articles[article['title']]={'content':article['content'],'urlToImage':article['urlToImage'],'publishedAt':article['publishedAt'],
                            'source':article['source']['name'],'url':article['url']}

# step3

#print('title : ',text_list[0].split('\n')[0])
        #step3
print('\t=====Trending Articles (Titles)======')
Trending_titles = list(Trending_articles.keys())
for value,title in enumerate(Trending_titles):
    print(value+1,':',title)
Trending_contents = []
Trending_image_urls = []
Trending_published_dates = []
Trending_sources = []
Trending_urls = []
for title in Trending_articles.keys():
    Trending_contents.append(str(Trending_articles[title]['content']))
    Trending_image_urls.append(str(Trending_articles[title]['urlToImage']))
    Trending_published_dates.append(str(Trending_articles[title]['publishedAt']))
    Trending_sources.append(str(Trending_articles[title]['source']))
    Trending_urls.append(str(Trending_articles[title]['url']))

for i in range(0,len(Trending_published_dates)):
    Trending_published_dates[i]=Trending_published_dates[i][:10]

Trending_news = [Trending_titles,Trending_contents, Trending_image_urls,Trending_published_dates,Trending_sources,Trending_urls]
encoded_contents = vectorizer.transform(Trending_contents)
print('\nr: Refresh List\n')
print('q: Quit()\n')


url = ("https://newsapi.org/v2/top-headlines?country=in&category="+ "Business"+"&apiKey=6b4e60b44b5746c3a5f75b18fccba442")
print('--------------------------------------------------------------------')
response = requests.get(url)
js = response.json()
Business_articles = defaultdict(dict)
for article in js['articles']:
    Business_articles[article['title']]={'content':article['content'],'urlToImage':article['urlToImage'],'publishedAt':article['publishedAt'],
                            'source':article['source']['name'],'url':article['url']}

Business_contents = []
Business_image_urls = []
Business_published_dates = []
Business_sources = []
Business_urls = []
Business_titles = list(Business_articles.keys())
for title in Business_articles.keys():
    Business_contents.append(str(Business_articles[title]['content']))
    Business_image_urls.append(str(Business_articles[title]['urlToImage']))
    Business_published_dates.append(str(Business_articles[title]['publishedAt']))
    Business_sources.append(str(Business_articles[title]['source']))
    Business_urls.append(str(Business_articles[title]['url']))
for i in range(0,len(Business_published_dates)):
    Business_published_dates[i]=Business_published_dates[i][:10]
Business_news = [Business_titles,Business_contents,Business_image_urls,Business_published_dates,Business_sources,Business_urls]

url = ("https://newsapi.org/v2/top-headlines?country=in&category="+ "Entertainment"+"&apiKey=6b4e60b44b5746c3a5f75b18fccba442")
response = requests.get(url)
js = response.json()
Entertainment_articles = defaultdict(dict)
for article in js['articles']:
    Entertainment_articles[article['title']]={'content':article['content'],'urlToImage':article['urlToImage'],'publishedAt':article['publishedAt'],
                            'source':article['source']['name'],'url':article['url']}

Entertainment_contents = []
Entertainment_image_urls = []
Entertainment_published_dates = []
Entertainment_sources = []
Entertainment_urls = []
Entertainment_titles = list(Entertainment_articles.keys())
for title in Entertainment_articles.keys():
    Entertainment_contents.append(str(Entertainment_articles[title]['content']))
    Entertainment_image_urls.append(str(Entertainment_articles[title]['urlToImage']))
    Entertainment_published_dates.append(str(Entertainment_articles[title]['publishedAt']))
    Entertainment_sources.append(str(Entertainment_articles[title]['source']))
    Entertainment_urls.append(str(Entertainment_articles[title]['url']))
for i in range(0,len(Entertainment_published_dates)):
    Entertainment_published_dates[i]=Entertainment_published_dates[i][:10]
Entertainment_news = [Entertainment_titles,Entertainment_contents,Entertainment_image_urls,Entertainment_published_dates,Entertainment_sources,Entertainment_urls]

url = ("https://newsapi.org/v2/top-headlines?country=in&category="
+ "sports"+"&apiKey=6b4e60b44b5746c3a5f75b18fccba442")
response = requests.get(url)
js = response.json()
Sports_articles = defaultdict(dict)
for article in js['articles']:
    Sports_articles[article['title']]={'content':article['content'],'urlToImage':article['urlToImage'],'publishedAt':article['publishedAt'],
                            'source':article['source']['name'],'url':article['url']}

Sports_contents = []
Sports_image_urls = []
Sports_published_dates = []
Sports_sources = []
Sports_urls = []
Sports_titles = list(Sports_articles.keys())
for title in Sports_articles.keys():
    Sports_contents.append(str(Sports_articles[title]['content']))
    Sports_image_urls.append(str(Sports_articles[title]['urlToImage']))
    Sports_published_dates.append(str(Sports_articles[title]['publishedAt']))
    Sports_sources.append(str(Sports_articles[title]['source']))
    Sports_urls.append(str(Sports_articles[title]['url']))
for i in range(0,len(Sports_published_dates)):
    Sports_published_dates[i]=Sports_published_dates[i][:10]

Sports_news = [Sports_titles,Sports_contents,Sports_image_urls,Sports_published_dates,Sports_sources,Sports_urls]

url = ("https://newsapi.org/v2/top-headlines?country=in&category="+ "Technology"+"&apiKey=6b4e60b44b5746c3a5f75b18fccba442")
response = requests.get(url)
js = response.json()
Technology_articles = defaultdict(dict)
for article in js['articles']:
    Technology_articles[article['title']]={'content':article['content'],'urlToImage':article['urlToImage'],'publishedAt':article['publishedAt'],
                            'source':article['source']['name'],'url':article['url']}

Technology_contents = []
Technology_image_urls = []
Technology_published_dates = []
Technology_sources = []
Technology_urls = []
Technology_titles = list(Technology_articles.keys())
for title in Technology_articles.keys():
    Technology_contents.append(str(Technology_articles[title]['content']))
    Technology_image_urls.append(str(Technology_articles[title]['urlToImage']))
    Technology_published_dates.append(str(Technology_articles[title]['publishedAt']))
    Technology_sources.append(str(Technology_articles[title]['source']))
    Technology_urls.append(str(Technology_articles[title]['url']))
for i in range(0,len(Technology_published_dates)):
    Technology_published_dates[i]=Technology_published_dates[i][:10]
Technology_news = [Technology_titles,Technology_contents,Technology_image_urls,Technology_published_dates,Technology_sources,Technology_urls]


# url = ("https://newsapi.org/v2/top-headlines?country=in&category="+ "general"+"&apiKey=6b4e60b44b5746c3a5f75b18fccba442")
# response = requests.get(url)
# js = response.json()

js = newsapi.get_everything(q='politics',
                                      sources='the-times-of-india',
                                      from_param=datetime.strftime(datetime.now() - timedelta(10), '%Y-%m-%d'),
                                      to=datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d'),
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)


Politics_articles = defaultdict(dict)
for article in js['articles']:
    Politics_articles[article['title']]={'content':article['content'],'urlToImage':article['urlToImage'],'publishedAt':article['publishedAt'],
                            'source':article['source']['name'],'url':article['url']}

Politics_contents = []
Politics_image_urls = []
Politics_published_dates = []
Politics_sources = []
Politics_urls = []
Politics_titles = list(Politics_articles.keys())
for title in Politics_articles.keys():
    Politics_contents.append(str(Politics_articles[title]['content']))
    Politics_image_urls.append(str(Politics_articles[title]['urlToImage']))
    Politics_published_dates.append(str(Politics_articles[title]['publishedAt']))
    Politics_sources.append(str(Politics_articles[title]['source']))
    Politics_urls.append(str(Politics_articles[title]['url']))
for i in range(0,len(Politics_published_dates)):
    Politics_published_dates[i]=Politics_published_dates[i][:10]
Politics_news = [Politics_titles,Politics_contents,Politics_image_urls,Politics_published_dates,Politics_sources,Politics_urls]




js = newsapi.get_top_headlines(sources='cnn, the-new-york-times')
World_articles=defaultdict(dict)
for article in js['articles']:
    World_articles[article['title']]={'content':article['content'],'urlToImage':article['urlToImage'],'publishedAt':article['publishedAt'],
                            'source':article['source']['name'],'url':article['url']}



World_contents = []
World_image_urls=[]
World_published_dates=[]
World_sources = []
World_urls = []
World_titles = list(World_articles.keys())
for title in World_articles.keys():
    World_contents.append(str(World_articles[title]['content']))
    World_image_urls.append(str(World_articles[title]['urlToImage']))
    World_published_dates.append(str(World_articles[title]['publishedAt']))
    World_sources.append(str(World_articles[title]['source']))
    World_urls.append(str(World_articles[title]['url']))
for i in range(0,len(World_published_dates)):
    World_published_dates[i]=World_published_dates[i][:10]
World_news = [World_titles,World_contents,World_image_urls,World_published_dates,World_sources,World_urls]


@app.route('/')
def index():
    return render_template('index.html',length=range(len(Trending_news[0])),Trending_news=Trending_news,Business_news=Business_news,Entertainment_news=Entertainment_news,Politics_news=Politics_news,Sports_news=Sports_news,Technology_news=Technology_news,World_news=World_news,world_length=range(len(range(10))))


@app.route('/<category>')
def category(category):
    if(category=='Business'):
        Category_news = Business_news;
    elif(category=='Sports'):
        Category_news = Sports_news
    elif(category=='Politics'):
        Category_news = Politics_news
    elif(category=='Technology'):
        Category_news=Technology_news
    elif(category == 'Entertainment'):
        Category_news = Entertainment_news
    elif(category =='World'):
        Category_news =World_news
    else:
        Category_news = Trending_news
    return render_template('category-post.html',category=category,Category_news=Category_news,Trending_news=Trending_news,Business_news=Business_news,Entertainment_news=Entertainment_news,Politics_news=Politics_news,Sports_news=Sports_news,Technology_news=Technology_news,World_news=World_news,world_length=range(10),length = range(len(Category_news[0])))

@app.route('/recommendation.html/<int:index>')
def recommendation(index):

    classifier_names = ['naive bayes','support vector machine','k-nearest neighbour','logistic regression','decision tree','randomforestclassifier']
    prediction_list = list();
    if(Trending_news[1][index]==None):
        content = str(Trending_news[0][index])
    else:
        content = str(Trending_news[1][index])
    for i in range(len(classifier_list)):
        print('\nClassifier name:',classifier_names[i])
        classifier = classifier_list[i]
        encoded_contents = vectorizer.transform([content])
        pred = classifier.predict(encoded_contents)
        prediction_list.append(pred[0]);
        print('predicted category: ',pred)

    prediction_list = Counter(prediction_list)
    print('prediction list:',prediction_list)
    category_index = prediction_list.most_common()[0][0]
    print('final predicted category: ',category_index)
    category = {0:'Business',1:'Entertainment',2:'Politics',3:'Sports',4:'Technology'}
    print('category of selected article: ',category[category_index],'\n')

    category = str(category[category_index])

    print('======================>Recommended articles<====================')
    if(category=='Politics'):
        content = Politics_news[1][index]
        Recom_articles = Politics_articles
    else:
        if(category=='Technology'):
            content = Technology_news[1][index]
            Recom_articles = Technology_articles
        elif(category=='Entertainment'):
            content = Entertainment_news[1][index]
            Recom_articles = Entertainment_articles
        elif(category=='Sports'):
            content = Sports_news[1][index]
            Recom_articles = Sports_articles
        elif(category=='Business'):
            content = Business_news[1][index]
            Recom_articles = Business_articles;
    Recom_contents = []
    Recom_image_urls=[]
    Recom_published_dates=[]
    Recom_sources = []
    Recom_urls = []
    Recom_titles = list(Recom_articles.keys())
    for title in Recom_articles.keys():
        Recom_contents.append(str(Recom_articles[title]['content']))
        Recom_image_urls.append(str(Recom_articles[title]['urlToImage']))
        Recom_published_dates.append(str(Recom_articles[title]['publishedAt']))
        Recom_sources.append(str(Recom_articles[title]['source']))
        Recom_urls.append(str(Recom_articles[title]['url']))
    for i in range(0,len(Recom_published_dates)):
        Recom_published_dates[i]=Recom_published_dates[i][:10]
    Recom_news = [Recom_titles,Recom_contents,Recom_image_urls,Recom_published_dates,Recom_sources,Recom_urls]


    stopWords = stopwords.words('english')
    Recom_content_set = list()
    for Recom_title in Recom_titles:
        Recom_content_set.append(str(Recom_articles[Recom_title]['content'])[:-20])

    tfidf_vectorizer = TfidfVectorizer(stop_words = stopWords)
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(iter(Recom_content_set))
    tfidf_matrix_test = tfidf_vectorizer.transform([content])

    similarity = cosine_similarity(tfidf_matrix_test,tfidf_matrix_train);
    print ("\nSimilarity Score [*] ",similarity)
    cos_sim = pd.Series(similarity[0])
    s_values = cos_sim.sort_values(ascending=False)
    s_indexes = cos_sim.sort_index(ascending=False)
    indexes = list(s_values.index)
    Selected_news = Trending_news
    values = list(s_values.values)
    print('indexes:',indexes)
    print('values: ',values)
    return render_template('recommendation.html',index=index, Selected_news=Selected_news, Recom_news=Recom_news,indexes=indexes,Trending_news=Trending_news,Business_news=Business_news,Entertainment_news=Entertainment_news,Politics_news=Politics_news,Sports_news=Sports_news,Technology_news=Technology_news,World_news=World_news,world_length=range(10))


@app.route('/recommendation.html/<string:category>/<int:index>')
def recommendation_category(category,index):

    # classifier_names = ['naive bayes','support vector machine','k-nearest neighbour','logistic regression','decision tree','randomforestclassifier']
    # prediction_list = list();
    # content = str(Trending_news[1][index])
    # for i in range(len(classifier_list)):
    #     print('\nClassifier name:',classifier_names[i])
    #     classifier = classifier_list[i]
    #     encoded_contents = vectorizer.transform([content])
    #     pred = classifier.predict(encoded_contents)
    #     prediction_list.append(pred[0]);
    #     print('predicted category: ',pred)
    #
    # prediction_list = Counter(prediction_list)
    # print('prediction list:',prediction_list)
    # category_index = prediction_list.most_common()[0][0]
    # print('final predicted category: ',category_index)
    # category = {0:'business',1:'entertainment',2:'politics',3:'sports',4:'technology'}
    # print('category of selected article: ',category[category_index],'\n')

    # str_category = str(category[category_index])

    print('======================>Recommended articles<====================')
    Selected_news=None
    if(category=='Trending'):
        classifier_names = ['naive bayes','support vector machine','k-nearest neighbour','logistic regression','decision tree','randomforestclassifier']
        prediction_list = list();
        # if(Trending_news[1][index]==None):
        #     content = str(Trending_news[0][index])
        # else:
        #     content = str(Trending_news[1][index])
        content = str(Trending_news[0][index])+'.' +str(Trending_news[1][index])
        for i in range(len(classifier_list)):
            print('\nClassifier name:',classifier_names[i])
            classifier = classifier_list[i]
            encoded_contents = vectorizer.transform([content])
            pred = classifier.predict(encoded_contents)
            prediction_list.append(pred[0]);
            print('predicted category: ',pred)

        prediction_list = Counter(prediction_list)
        print('prediction list:',prediction_list)
        category_index = prediction_list.most_common()[0][0]
        print('final predicted category: ',category_index)
        category = {0:'Business',1:'Entertainment',2:'Politics',3:'Sports',4:'Technology'}
        print('category of selected article: ',category[category_index],'\n')

        category = str(category[category_index])
        Selected_news=Trending_news
    if(category=='Politics'):
        # if(Politics_news[1][index]==None):
        #     content = Politics_news[0][index]
        # else:
        #     content = Politics_news[1][index]
        content = str(Trending_news[0][index])+'.' +str(Trending_news[1][index])
        if(Selected_news==None):
            Selected_news=Politics_news
        Recom_news = Politics_news
    elif(category=='World'):
        # if(World_news[1][index]==None):
        #     content = World_news[0][index]
        # else:
        #     content = World_news[1][index]
        content = str(Trending_news[0][index])+'.' +str(Trending_news[1][index])
        if(Selected_news==None):
            Selected_news = World_news
        Recom_news = World_news
    else:
        if(category=='Technology'):
            # if(Technology_news[1][index]==None):
            #     content = Technology_news[0][index]
            # else:
            #     content = Technology_news[1][index]
            content = str(Trending_news[0][index])+'.' +str(Trending_news[1][index])
            if(Selected_news==None):
                Selected_news = Technology_news
            Recom_news = Technology_news
        elif(category=='Entertainment'):
            # if(Entertainment_news[1][index]==None):
            #     content = Entertainment_news[0][index]
            # else:
            #     content = Entertainment_news[1][index]
            content = str(Trending_news[0][index])+'.' +str(Trending_news[1][index])
            if(Selected_news==None):
                Selected_news = Entertainment_news
            Recom_news = Entertainment_news
        elif(category=='Sports'):
            # if(Sports_news[1][index]==None):
            #     content = Sports_news[0][index]
            # else:
            #     content = Sports_news[1][index]
            content = str(Trending_news[0][index])+'.' +str(Trending_news[1][index])
            if(Selected_news==None):
                Selected_news = Sports_news
            Recom_news = Sports_news
        elif(category=='Business'):
            # if(Business_news[1][index]==None):
            #     content = Business_news[0][index]
            # else:
            #     content = Business_news[1][index]
            content = str(Trending_news[0][index])+'.' +str(Trending_news[1][index])
            if(Selected_news==None):
                Selected_news = Business_news
            Recom_news = Business_news;

    # Recom_contents = []
    # Recom_image_urls=[]
    # Recom_published_dates=[]
    # Recom_sources = []
    # Recom_urls = []
    # Recom_titles = list(Recom_articles.keys())
    # for title in Recom_articles.keys():
    #     Recom_contents.append(str(Recom_articles[title]['content']))
    #     Recom_image_urls.append(str(Recom_articles[title]['urlToImage']))
    #     Recom_published_dates.append(str(Recom_articles[title]['publishedAt']))
    #     Recom_sources.append(str(Recom_articles[title]['source']))
    #     Recom_urls.append(str(Recom_articles[title]['url']))
    # for i in range(0,len(Recom_published_dates)):
    #     Recom_published_dates[i]=Recom_published_dates[i][:10]
    # Recom_news = [Recom_titles,Recom_contents,Recom_image_urls,Recom_published_dates,Recom_sources,Recom_urls]


    stopWords = stopwords.words('english')
    Recom_content_set = list()
    for i in range(len(Recom_news[0])):
        Recom_content_set.append(str(Recom_news[1][i])[:-20])

    tfidf_vectorizer = TfidfVectorizer(stop_words = stopWords)
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(iter(Recom_content_set))
    tfidf_matrix_test = tfidf_vectorizer.transform([content])

    similarity = cosine_similarity(tfidf_matrix_test,tfidf_matrix_train);
    print ("\nSimilarity Score [*] ",similarity)
    cos_sim = pd.Series(similarity[0])
    s_values = cos_sim.sort_values(ascending=False)
    s_indexes = cos_sim.sort_index(ascending=False)
    indexes = list(s_values.index)
    indexes.remove(index)
    values = list(s_values.values)
    print('indexes:',indexes)
    print('values: ',values)


    return render_template('recommendation.html',category=category, index=index, Selected_news=Selected_news, Recom_news=Recom_news,indexes=indexes,Trending_news=Trending_news,Business_news=Business_news,Entertainment_news=Entertainment_news,Politics_news=Politics_news,Sports_news=Sports_news,Technology_news=Technology_news,World_news=World_news,world_length=range(10))

@app.route('/contact.html')
def contact():
    return render_template('contact.html')
