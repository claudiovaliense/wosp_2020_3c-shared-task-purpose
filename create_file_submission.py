"""
Autor: Claudio Moisés Valiense de Andrade
Objetivo: Criar um biblioteca de funções de uso geral
"""

import csv  # Manipular csv
import json  # Manipulate extension json

def load_dict_file(file):
    """Load dict in file"""
    with open(file, 'r', newline='') as csv_reader:
        return json.load(csv_reader)
    
def file_submit_kaggle3(dataset, classificador):
    
    result = load_dict_file('y_pred/' +dataset +'_'+classificador +'_f1_macro_test0')        
    saida = open('dataset/' +dataset +'/kaggle.csv', 'w')
    
    filename = 'dataset/' +dataset +'/test.csv'
    file = csv.reader(open(filename))
    next(file) # pula cabecalho
    id = []
    for line in file:
        id.append(line[0])
        
    saida.write("unique_id,citation_class_label\n")
    index_id = 0
    for v in result['y_pred-folds']['0']:
        saida.write(id[index_id] +"," +str(int(v)) +"\n")        
        index_id+=1


file_submit_kaggle3('3c-shared-task-purpose', 'passive_aggressive')
