# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 00:55:37 2018

@author: Biswadeep Sen
"""

import ast

#f = open("intruder_word_list.txt","r")
g = open("ground_truth.txt","r")
#h = open("intruder(0.01,0.01)[2].txt","r")
h = open("intruder(0.1,0.1).txt","r")
#h = open("LDA100_modified.txt","r")
#h = open("LDA200_modified(samegroup).txt","r")

gt = g.readlines()
algo_output = h.readlines()

gt = [ast.literal_eval(i) for i in gt if len(i.strip()) > 0]
algo_output = [(j) for j in algo_output if len(j.strip()) > 0]
algo_output = [ast.literal_eval(j) for j in algo_output]

intruders = {}

#g = f.readlines()
#
#for i in g:
#    i = i.strip()
#    k = i.split()
#    intruders[k[0]] = (1/int(k[1]))
    
def similar(L1,L2):
    sim = 0
    for i in L1:
        if i in L2:
            sim += 1
    return (sim)

score = 0

for i in gt:
    #print(i)
    for j in algo_output:
        #print("j " + str(j))
        if i[0] == j[0] and i[4] == j[4]:
            print(i,j)
            #print(score)
            score += similar(i[1:4],j[1:4])*i[5]

#for j in algo_output:
    #print("j " + str(j))                   

print("score =" + str(score))









