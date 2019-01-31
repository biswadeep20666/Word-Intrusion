# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:52:27 2018

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

#new_model = gensim.models.KeyedVectors.load_word2vec_format('en_wiki_vectors.bin',binary = True)

seen = {}

#def word_doc_relations(word):
#    docs = []
#    with open("DB_corpus3.txt", 'r', encoding = 'utf-8') as infile:
#        for line in infile:
#            line = line.strip() #Stripping the line of trailing spaces
#            k = line.split('\t') #splitting title and body of the documents
#            #if word.lower() in k[1].lower(): #If the word occurs in the document
#            if k[1].lower().count(' ' + word.lower() + ' ') > 4:
#                docs.append((k[0],"d")) #append the doc to the list of docs where the word occurs
#        return docs #return the set of docs as list

#def word_doc_relations(word):
#    docs = []
#    with open("DB_corpus3.txt", 'r', encoding = 'utf-8') as infile:
#        for line in infile:
#            line = line.strip() #Stripping the line of trailing spaces
#            k = line.split('\t') #splitting title and body of the documents
#            #if word.lower() in k[1].lower(): #If the word occurs in the document
#            abstract = k[1].lower()
#            words = word_tokenize(abstract)
#            if words.count(word.lower()) > 4:
#                docs.append(k) #append the doc to the list of docs where the word occurs
#        if docs == []:#If the word is rarely occuring in a document
#            print("rare!")
#            with open("DB_corpus3.txt", 'r', encoding = 'utf-8') as infile:
#                for line in infile:
#                    line = line.strip() #Stripping the line of trailing spaces
#                    k = line.split('\t') #splitting title and body of the documents
#                    #if word.lower() in k[1].lower(): #If the word occurs in the document
#                    abstract = k[1].lower()
#                    words = word_tokenize(abstract)
#                    if words.count(word.lower()) > 4:
#                        docs.append(k)
#        titles =  [i[0] for i in docs]#List containing titles of the documents
#        word_in_doc = [word_tokenize(i[1]) for i in docs]
#        total_doc_len = sum([len(i) for i in word_in_doc])
#        prob_ratios = [((i.count(word)*len(i))/total_doc_len)  for i in word_in_doc]
#        normalizer = sum(prob_ratios)
#        prob = [(i/normalizer) for i in prob_ratios]
#        print(titles)
#        return (np.random.choice(titles, p = prob), "d")  #return the set of docs as list

def doc_contains_something(doc):
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

#def doc_word_relations(doc):
#    words = [] #List containing related words with identifier
#    with open("DB_corpus3.txt", 'r', encoding = 'utf-8') as infile:
#        for line in infile:
#            line = line.strip() #Splitting the document of trailing spaces
#            k = line.split('\t')#Splitting the title and the body of the documents
#            if k[0].lower() == doc.lower():#If the title matches we return all the words of the document
#                words = word_tokenize(k[1]) #Tokenizing the sentence into words
#                Dict = nltk.FreqDist(words)
#                most_common = Dict.most_common(10)
#                print(most_common)
#                probability_proportions = [i[1] for i in most_common]#Calculaing the probability of taking each word according to it's number of occurence in the document
#                common_words = [i[0] for i in most_common]
#                normalizer = sum(probability_proportions)
#                probabilities = [(i/normalizer) for i in probability_proportions]
#                float_extra = 1 - sum(probabilities)
#                probabilities[0] += float_extra
#                #print(probabilities)
#                chosen_word = np.random.choice(common_words,p = probabilities)
#        return (chosen_word,"w")
    
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
    
#def word_intrusion_generator(pivot):
#    data = [pivot]
#    nbhrs = word_doc_relations(pivot) + word_word_relations(pivot)  #Calculating the
#    chosen_nbhr = random.choice(nbhrs)
#    while len(data) < 4:
#        if chosen_nbhr[1] == "w":
#            print("w = " + chosen_nbhr[0])
#            k = random.random() #Choosing a parameter between 
#            if chosen_nbhr[1] == "w" and k > 0.1:
#                print(k)
#                print("Selected_word = " + chosen_nbhr[0])
#                data.append(chosen_nbhr[0])
#            pivot = chosen_nbhr[0]
#            nbhrs = word_doc_relations(pivot)  #Calculating the
#            print("word nbhrs = "+ str(nbhrs))
#            chosen_nbhr = random.choice(nbhrs)
#        elif chosen_nbhr[1] == 'd':
#            print("d = " + chosen_nbhr[0])
#            nbhrs = doc_word_relations(chosen_nbhr[0]) + doc_doc_relations(chosen_nbhr[0])
#            print("doc nbhrs = "+ str(nbhrs))
#            chosen_nbhr = random.choice(nbhrs)
#    return data

#def intruder(pivot):
#    data = []
#    r = random.random()
#    if r > 0.5:
#        chosen_nbhr = word_doc_relations(pivot)#Calculating the document neighbors of the word
#    else:
#        chosen_nbhr = word_word_relations(pivot)
#    while len(data) < 1:
#        if chosen_nbhr[1] == "w":#If the chosen neighbor is a word
#            print("w = " + chosen_nbhr[0])
#            k = random.random() #Choosing a parameter between 
#            if chosen_nbhr[1] == "w" and k < 0.4 and chosen_nbhr[0] not in data: #with probability 0.9 we accept the word into our dataset
#                print(k)
#                print("Selected_word = " + chosen_nbhr[0])
#                data.append(chosen_nbhr[0])
#            pivot = chosen_nbhr[0]#This neighbor becomes the new pivot
#            r = random.random()
#            if r < 0.9:
#                chosen_nbhr = word_doc_relations(pivot)#Calculating the document neighbors of the word
#            else:
#                chosen_nbhr = word_word_relations(pivot)  #Calculating the word neighbors

            
#        elif chosen_nbhr[1] == 'd':
#            print("d = " + chosen_nbhr[0])
#            r = random.random()
#            if r > 0.1:
#                chosen_nbhr = doc_word_relations(chosen_nbhr[0]) 
#            else:
#                chosen_nbhr = doc_doc_relations(chosen_nbhr[0])
            
#    return data   
    
            
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
            if chosen_nbhr[1] == "w" and k < 0.9 and (chosen_nbhr[0]).lower() not in data and ps.stem((chosen_nbhr[0]).lower()) not in data and lemmatizer.lemmatize(chosen_nbhr[0]) not in data and M >= 1: #with probability 0.9 we accept the word into our dataset
                #print(k)
                print("Selected_word = " + chosen_nbhr[0])
                i.append(chosen_nbhr[0])
            pivot = chosen_nbhr[0]#This neighbor becomes the new pivot
            r = random.random()
            if r < 1 and word_word_relations(pivot) != []:
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



g = open("word_word_only.txt",'w',encoding = 'utf-8')

g.write(str(word_intrusion_generator("chess"))+'\n')
g.write(str(word_intrusion_generator("football"))+'\n')
g.write(str(word_intrusion_generator("alps"))+'\n')
g.write(str(word_intrusion_generator("euler"))+'\n')
g.write(str(word_intrusion_generator("ibm"))+'\n')
g.write(str(word_intrusion_generator("hitchcock"))+'\n')
g.write(str(word_intrusion_generator("compiler"))+'\n')
g.write(str(word_intrusion_generator("skype"))+'\n')
g.write(str(word_intrusion_generator("linux"))+'\n')
g.write(str(word_intrusion_generator("forest"))+'\n')
g.write(str(word_intrusion_generator("swimming"))+'\n')



g.write(str(word_intrusion_generator("computer"))+'\n')
g.write(str(word_intrusion_generator("car"))+'\n')
g.write(str(word_intrusion_generator("chair"))+'\n')
g.write(str(word_intrusion_generator("canning"))+'\n')
g.write(str(word_intrusion_generator("cult"))+'\n')
g.write(str(word_intrusion_generator("courtesy"))+'\n')
g.write(str(word_intrusion_generator("cat"))+'\n')
g.write(str(word_intrusion_generator("catamaran"))+'\n')
g.write(str(word_intrusion_generator("cow"))+'\n')
g.write(str(word_intrusion_generator("continuous"))+'\n')


g.write(str(word_intrusion_generator("fan"))+'\n')
g.write(str(word_intrusion_generator("throne"))+'\n')
g.write(str(word_intrusion_generator("sword"))+'\n')
g.write(str(word_intrusion_generator("arrow"))+'\n')
g.write(str(word_intrusion_generator("shield"))+'\n')
g.write(str(word_intrusion_generator("armor"))+'\n')
g.write(str(word_intrusion_generator("crown"))+'\n')
g.write(str(word_intrusion_generator("bow"))+'\n')
g.write(str(word_intrusion_generator("knife"))+'\n')
g.write(str(word_intrusion_generator("king"))+'\n')



g.write(str(word_intrusion_generator("shoulder"))+'\n')
g.write(str(word_intrusion_generator("table"))+'\n')
g.write(str(word_intrusion_generator("box"))+'\n')
g.write(str(word_intrusion_generator("light"))+'\n')
g.write(str(word_intrusion_generator("switch"))+'\n')
g.write(str(word_intrusion_generator("cover"))+'\n')
g.write(str(word_intrusion_generator("window"))+'\n')
g.write(str(word_intrusion_generator("wicket"))+'\n')
g.write(str(word_intrusion_generator("screw"))+'\n')
g.write(str(word_intrusion_generator("gun"))+'\n')



g.write(str(word_intrusion_generator("shuttlecock"))+'\n')
g.write(str(word_intrusion_generator("doraemon"))+'\n')
g.write(str(word_intrusion_generator("racket"))+'\n')
g.write(str(word_intrusion_generator("tennis"))+'\n')
g.write(str(word_intrusion_generator("anime"))+'\n')
g.write(str(word_intrusion_generator("wire"))+'\n')
g.write(str(word_intrusion_generator("thunderbird"))+'\n')
g.write(str(word_intrusion_generator("clock"))+'\n')
g.write(str(word_intrusion_generator("shadow"))+'\n')
g.write(str(word_intrusion_generator("finch"))+'\n')

g.close()

#
#
##print(word_doc_relations("scorpion"))    
##print(word_intrusion_generator("immortal"))
##print(word_intrusion_generator("opeth"))
##print(word_intrusion_generator("vampire"))
##print(word_intrusion_generator("ibm"))
##print(word_intrusion_generator("Galois"))  
##print(word_intrusion_generator("Turing")) 
##print(word_doc_relations("scorpion"))    
#    
#g.close()
#
#
##print(str(word_intrusion_generator("football")))
#
#
##print(word_intrusion_generator("football"))
##print(word_intrusion_generator("alps"))
##print(word_intrusion_generator("ibm"))
##print(word_intrusion_generator("hitchcock"))
##print(word_intrusion_generator("compiler"))
##print(word_intrusion_generator("skype"))
##print(word_intrusion_generator("linux"))
##print(word_intrusion_generator("forest"))
##print(word_intrusion_generator("swimming"))
#    
##word_doc_relations("hyblaean")
#
#
#
