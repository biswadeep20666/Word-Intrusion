# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 06:33:02 2019

@author: Biswadeep Sen
"""

import math

import gensim

import random

import numpy as np

from gensim.models import Word2Vec

from nltk.stem import SnowballStemmer

ps = SnowballStemmer("english")

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

new_model = gensim.models.KeyedVectors.load_word2vec_format('DB_vectors3.bin',binary = True)

g = open("IDF.txt",'r',encoding = 'utf-8')

k = g.readlines()

IDF = {}

for i in k:
    j = i.split('\t')
    IDF[j[0]] = float(j[1])

#def tf(word, doc):
#    return (doc.split()).count(word) / len(doc.split())
    
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def tf(word, doc):
    return doc.split().count(word)

def imp_words2(doc):
    scores = {}
#    doc_len = len(doc.split())
    #print("doc len is: " + str(doc_len))
    for word in (doc.split()):
#        print(word)
#        print("tf= "+str(tf(word,doc)))
#        print("log idf= " + str(math.log(IDF[word])))
        #scores[word] = math.log(1+(l/(1-l))*(tf(word,doc)/doc_len)*IDF[word])
        if IDF[word] > 1000 and IDF[word] < 19815190 and len(word) >= 3 and is_ascii(word):
            scores[word] = tf(word,doc)*math.log(IDF[word])
        #print(scores[word])
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_words == []:
        return []
    return sorted_words[:7]


seen = {}



def doc_contains_something(doc):
    """Returns True of the content of the document is nom-empty"""
    with open("DB_corpus4.txt", 'r', encoding = 'utf-8') as infile:
        for line in infile:
            line = line.strip() #Splitting the document of trailing spaces
            k = line.split('\t')
            if len(k) > 2:
                return True
            else:
                return False
    return False     

def word_doc_relations(word):
    """Takes a word as input and from all the documents containing that word returns one document probabilistically with probabilities proportionate to tf_idf scores"""
    if word in seen:
        names = seen[word][0]
        probabilities = seen[word][1]
    else:
        tf_idf_word_in_doc = {}
        doc_len = {}
        with open("DB_corpus4.txt", 'r', encoding = 'utf-8') as infile:
            for line in infile:
                line = line.strip() #Stripping the line of trailing spaces
                k = line.split('\t') #splitting title and body of the documents
                
                if len(k) == 2 and word in (k[1].split()):
                    #print(line)
                    doc_len[k[0]] = len((k[1]).split())
                    #tf_idf_word_in_doc[k[0]] = math.log(1+(l/(1-l))*(tf(word,k[1])/doc_len[k[0]])*IDF[word])
                    tf_idf_word_in_doc[k[0]] = tf(word,k[1])*math.log(IDF[word])
                    #print("tfidf value is: " + str(tf_idf_word_in_doc[k[0]]))   
                    
                    #print("len is: " + str(doc_len[k[0]]))
        #print("Reached line 102!")
        total_len = 0
        for i in doc_len:
            total_len += doc_len[i]
        #print("Total len" + str(total_len))
        for i in doc_len:
            doc_len[i] = (doc_len[i]/total_len)
            #print("len ratio: " + str(doc_len[i])) 
        prob_portions = {}
        for i in tf_idf_word_in_doc:
            prob_portions[i] = tf_idf_word_in_doc[i]*doc_len[i]
            #print("Doc is: " + i + "prob portion is: " + str(prob_portions[i]))
        top_docs = sorted(prob_portions.items(), key=lambda x: x[1], reverse=True)[:10]
        #print(top_docs)
        names = []
        prob_portions = []
        for i in top_docs:
            names.append(i[0])
            prob_portions.append(i[1])
            normalizer = sum(prob_portions)
            probabilities = [(i/normalizer) for i in prob_portions]
        seen[word] = (names,probabilities)	        
    return (np.random.choice(names,p = probabilities),"d")


def doc_word_relations(doc):
    """Takes a document as input and among all words occuring in the document returns a word with high tf_idf score randomly with probabilities proportionate to tf-idf scores"""
    with open("DB_corpus4.txt", 'r', encoding = 'utf-8') as infile:
        for line in infile:
            line = line.strip() #Splitting the document of trailing spaces
            k = line.split('\t')#Splitting the title and the body of the documents
            if k[0].lower() == doc.lower():#If the title matches we return all the words of the document
                imp_words = imp_words2(k[1]) 
                #print(imp_words)
                if imp_words == []:
                    return []
                probability_proportions = [i[1] for i in imp_words]#Calculaing the probability of taking each word according to it's number of occurence in the document
                common_words = [i[0] for i in imp_words]
                normalizer = sum(probability_proportions)
                probabilities = [(i/normalizer) for i in probability_proportions]
                chosen_word = np.random.choice(common_words,p = probabilities)
        return (chosen_word,"w")

def word_word_relations(word): #This will give closest words judging by word document relations
    """Takes a word as input and among the words closest to that word in the word embedding representation it returns another word with probability proportionate to cosine distance"""
    similar_words = new_model.most_similar(word,topn=5)#Returns similar words by calculating cosine similarity
    similar_words = [i for i in similar_words if is_ascii(i[0]) and IDF[i[0]] < 19815190 and IDF[i[0]] > 2000]
    cos_dist = [(1-i[1]) for i in similar_words] #list containing the word and it's cosine distance from the pivot
    similar_words3 = [(i[0]) for i in similar_words]
    probability_proportions = [math.exp(-j*j) for j in cos_dist]#theratio in which the probailities of the numbers are to be taken
    normalizer = sum(probability_proportions)
    probabilities = [(i/normalizer) for i in probability_proportions]
#    float_extra = 1 - sum(probabilities)
#    probabilities[0] += float_extra
    if similar_words3 == []:
        return []
    chosen_word = np.random.choice(similar_words3, p = probabilities)
    return (chosen_word,"w")

def doc_doc_relations(doc,lambd = -1):    #This will return TF-IDF similar doc and hyperlinks connected docs
    """Takes a document as input and from all documents that has a hyperlink from that document it randomly chooses and returns one"""
    r = random.random()
    hyperlink_relations = []
    if r > lambd:
        with open("hyperlink_information.txt",'r',encoding = 'utf-8') as infile:
            for line in infile:
                line = line.strip()
                docs = line.split('\t')#Extracting all the document relations by hyperlink distance
                if docs[0].lower() == doc.lower():#If their title matches
                    k = docs[1:]
                    hyperlink_relations = []
                    for i in k:
                        if doc_contains_something(i):
                            hyperlink_relations.append((i,"d"))
    if len(hyperlink_relations) > 0:
        return random.choice(hyperlink_relations)
    else:
        return []
    

def word_intrusion_generator(pivot): #Generating the random walk
    pivot = pivot.lower()
    data = [pivot]
    r = random.random()
    if r > 0.99:
    #if r > 2:
        chosen_nbhr = word_doc_relations(pivot)#Calculating the document neighbors of the word
    else:
        chosen_nbhr = word_word_relations(pivot)
    while len(data) < 4:
        if chosen_nbhr[1] == "w":#If the chosen neighbor is a word
            print("w = " + chosen_nbhr[0])
            k = random.random() #Choosing a parameter between 
            if chosen_nbhr[1] == "w" and k < 0.9 and lemmatizer.lemmatize(chosen_nbhr[0]) not in data and (ps.stem(chosen_nbhr[0]).lower()) not in data and (chosen_nbhr[0]).lower() not in data and (ps.stem(chosen_nbhr[0]).lower()) not in data[-1].lower() and IDF[chosen_nbhr[0]] > 1000 and IDF[chosen_nbhr[0]] < 198151900: #with probability 0.9 we accept the word into our dataset
                print(k)
                print("Selected_word = " + chosen_nbhr[0].lower())
                data.append(chosen_nbhr[0].lower())
                print(data)
            pivot = chosen_nbhr[0]#This neighbor becomes the new pivot
            r = random.random()
            if r < 0.99 and word_word_relations(pivot) != []:#If r is less than the given value choose word_word relations else choose word document
                chosen_nbhr = word_word_relations(pivot)#Calculating the document neighbors of the word
            else:
                chosen_nbhr = word_doc_relations(pivot)  #Calculating the word neighbors
            
        elif chosen_nbhr[1] == 'd':
            print("d = " + chosen_nbhr[0])
            k = random.random()
            if k < 0.9 and len(chosen_nbhr[0].split()) <= 3 and ps.stem(chosen_nbhr[0].lower()) not in data:
                data.append(chosen_nbhr[0].lower())   
                print(data)
            r = random.random()
            if r < 0.1/(-len(data)):#If r is less than the given parameter choose doc_doc relations
                chosen_nbhr = doc_doc_relations(chosen_nbhr[0]) 
            else:
                chosen_nbhr = doc_word_relations(chosen_nbhr[0])
    i = []
    M = 0
    #M = 1
    while len(i) < 1:
        
        if chosen_nbhr[1] == "w":#If the chosen neighbor is a word
            print("w = " + chosen_nbhr[0])
            k = random.random() #Choosing a parameter between 
            #print("k is: " + str(k))
            #print("M is: " + str(M))
            if chosen_nbhr[1] == "w" and k < 0.8 and (chosen_nbhr[0]).lower() not in data and ps.stem((chosen_nbhr[0]).lower()) not in data and lemmatizer.lemmatize(chosen_nbhr[0]) not in data and M >= 1: #with probability 0.9 we accept the word into our dataset
                #print(k)
                print("Selected_word = " + chosen_nbhr[0])
                i.append(chosen_nbhr[0])
            pivot = chosen_nbhr[0]#This neighbor becomes the new pivot
            r = random.random()
            if r < 0.5 and word_word_relations(pivot) != []:
                chosen_nbhr = word_word_relations(pivot)#Calculating the document neighbors of the word
            else:
                chosen_nbhr = word_doc_relations(pivot)  #Calculating the word neighbors
            
        elif chosen_nbhr[1] == 'd':
            print("d = " + chosen_nbhr[0])
            r = random.random()
            if r < 0.9 and doc_doc_relations(chosen_nbhr[0]) != [] and M < 1:
                M += 1
                chosen_nbhr = doc_doc_relations(chosen_nbhr[0]) 
            else:
                M += 1
                chosen_nbhr = doc_word_relations(chosen_nbhr[0])
                
    return data + i   

print(word_intrusion_generator("football"))