"""
Autor: Claudio Moisés Valiense de Andrade
"""

import csv  # Manipular csv

def create_text():
    dataset = '3c-shared-task-purpose'    
    filename = 'dataset/'+dataset +'/train.csv'
    file = csv.reader(open(filename))    
    texts = open('dataset/'+dataset+"/orig/texts.txt", "w")
    score = open('dataset/'+dataset+"/orig/score.txt", "w")
    next(file)    
    
    for field in file:           
        text_agg=[]
        text_agg.append(field[2])
        text_agg.append(field[4])
        text_agg.append(field[6])             
        text_agg = " ".join(text_agg)        
        texts.write(text_agg.strip() +"\n")    
        score.write(field[7] +"\n")        


create_text()