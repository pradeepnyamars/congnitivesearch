# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:45:38 2019

@author: pnyamars
"""
import re
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from docx import Document


def tf(word, doc):
    return doc.count(word) / len(doc)

def n_containing(word, doclist):
    return sum(1 for doc in doclist if word in doc)

def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)))

def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))

def search_words(searchsentence):
    try:
        # split sentence into individual words 
        searchsentence = searchsentence.lower()
        try:
            words = searchsentence.split(' ')
        except:
            words = list(words)
        enddic = {}
        idfdic = {}
       
        
        # remove words if not in worddic 
        realwords = []
        for word in words:
            if word in list(worddic.keys()):
                realwords.append(word)  
        words = realwords
        numwords = len(words)
        
        # make metric of number of occurances of all words in each doc & largest total IDF 
        for word in words:
            for indpos in worddic[word]:
                index = indpos[0]
                amount = len(indpos[1])
                idfscore = indpos[2]
                enddic[index] = amount
                idfdic[index] = idfscore
                fullcount_order = sorted(enddic.items(), key=lambda x:x[1], reverse=True)
                fullidf_order = sorted(idfdic.items(), key=lambda x:x[1], reverse=True)

                
        # make metric of what percentage of words appear in each doc
        combo = []
        alloptions = {k: worddic.get(k, None) for k in (words)}
        for worddex in list(alloptions.values()):
            for indexpos in worddex:
                for indexz in indexpos:
                    combo.append(indexz)
        comboindex = combo[::3]
        combocount = Counter(comboindex)
        for key in combocount:
            combocount[key] = combocount[key] / numwords
        combocount_order = sorted(combocount.items(), key=lambda x:x[1], reverse=True)
        """
        # make metric for if words appear in same order as in search
        if len(words) > 1:
            x = []
            y = []
            for record in [worddic[z] for z in words]:
                for index in record:
                     x.append(index[0])
            for i in x:
                if x.count(i) > 1:
                    y.append(i)
            y = list(set(y))

            closedic = {}
            for wordbig in [worddic[x] for x in words]:
                for record in wordbig:
                    if record[0] in y:
                        index = record[0]
                        positions = record[1]
                        try:
                            closedic[index].append(positions)
                        except:
                            closedic[index] = []
                            closedic[index].append(positions)

            x = 0
            fdic = {}
            for index in y:
                csum = []
                for seqlist in closedic[index]:
                    while x > 0:
                        secondlist = seqlist
                        x = 0
                        sol = [1 for i in firstlist if i + 1 in secondlist]
                        csum.append(sol)
                        fsum = [item for sublist in csum for item in sublist]
                        fsum = sum(fsum)
                        fdic[index] = fsum
                        fdic_order = sorted(fdic.items(), key=lambda x:x[1], reverse=True)
                    while x == 0:
                        firstlist = seqlist
                        x = x + 1
        else:
            fdic_order = 0"""                  
                
        
        return(searchsentence,words,fullcount_order,combocount_order,fullidf_order)
    
    except:
        return("")
        
def search(term):
    results = search_words(term)

    # get metrics 
    num_score = results[2]
    per_score = results[3]
    tfscore = results[4]
  
    
    final_candidates = []

   
    try:
        first_candidates = []

        second_candidates = []

     
    # rule2: next add other word order score which are greater than 1 

        t3_order = first_candidates[0:3]
        for each in t3_order:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    # rule3: next add top td-idf results
        final_candidates.insert(len(final_candidates),tfscore[0][0])
        final_candidates.insert(len(final_candidates),tfscore[1][0])

    # rule4: next add other high percentage score 
        t3_per = second_candidates[0:3]
        for each in t3_per:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    #rule5: next add any other top results for metrics
        othertops = [num_score[0][0],per_score[0][0],tfscore[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)
                
    # unless single term searched, in which case just return 
    except:
        othertops = [num_score[0][0],num_score[1][0],num_score[2][0],per_score[0][0],tfscore[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)

   
    print(final_candidates)
    for index, results in enumerate(final_candidates):
        if index < 5:
            print("RESULT", index + 1, ":", alldocslist[results])        
        


#filename = 'CAEUSER1.docx'
filename = 'CAE Wiki Pages1.docx'
file_ref = open(filename,"rb")
doc = Document(file_ref)

sentences=[]
for para in doc.paragraphs:
    sentences.append(para.text) 
    
# pre processing    
alldocslist =[]
for sentence in sentences:
    sentence = sentence.lower()
    clean_text = re.sub(r'\W',' ',sentence)
    clean_text = re.sub(r'\s+\d+\s', ' ',clean_text)
    clean_text = re.sub(r'\d+\W+\s', ' ',clean_text)
    clean_text = re.sub(r'\s+[a-z]\s+',' ',clean_text)
    clean_text = re.sub(r'\s+',' ',clean_text)
    if len(clean_text.strip())!=0:
        alldocslist.append(clean_text) 
    # remove stop words
    #word_list= [word for word in clean_text.split() if word not in stopwords.words('english')]
    #clean_text= ' '.join(word_list)
       
plot_data = [[]]*len(alldocslist)
index=0
for doc in alldocslist:
    tokentext = [word for word in word_tokenize(doc) if word not in stopwords.words('english')] 
    plot_data[index].append(tokentext)
    index+=1
    
   
# Create inverse index which gives document number for each document and where word appears
l = plot_data[0]
flatten = [item for sublist in l for item in sublist]
words = flatten
wordsunique = set(words)
wordsunique = list(wordsunique)


plottest = plot_data[0][0:1000]

worddic = {}

for doc in plottest:
    for word in wordsunique:
        if word in doc:
            word = str(word)
            index = plottest.index(doc)
            positions = list(np.where(np.array(plottest[index]) == word)[0])
            idfs = tfidf(word,doc,plottest)
            try:
                worddic[word].append([index,positions,idfs])
            except:
                worddic[word] = []
                worddic[word].append([index,positions,idfs])
                
                
search("cae migration process")                