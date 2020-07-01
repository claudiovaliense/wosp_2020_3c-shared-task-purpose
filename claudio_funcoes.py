"""
Autor: Claudio Moisés Valiense de Andrade
Objetivo: Criar um biblioteca de funções de uso geral
"""

import csv  # Manipular csv
import numpy  # Manipular arrau numpy
import scipy.stats as stats # Calcular intervalo de confiança
import math # Calcular ndcg
numpy.random.seed(seed=42)

from sklearn.decomposition import LatentDirichletAllocation  # LDA
import timeit  # calcular metrica de tempo
import pickle  # Usado para salvar modelo
import joblib  # Usado para salvar modelo
from sklearn.feature_extraction.text import TfidfVectorizer  # Utilizar metodo Tfidf
import os  # Variable in system
from datetime import datetime  # Datetime for time in file
from heapq import nlargest  # Order vector
import json  # Manipulate extension json
import statistics
import pandas as pd # matrix confusion
from sklearn.metrics import confusion_matrix
import sklearn.metrics
import heapq_max
import re # Regular expression
import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize 
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_svmlight_file, load_svmlight_files
import ast # String in dict
from sklearn import svm  # Classifier SVN
import sys
import io
import smtplib
from email.mime.text import MIMEText
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import psycopg2 # conectar postgresql
import unidecode # remove accents
import urllib.parse #url encode
from pyjarowinkler import distance # similarity string per caracters
import imblearn # under and over sample
#import distance


numpy.set_printoptions(threshold=sys.maxsize)

# tunned params
#C_lbd = numpy.append(2.0 ** numpy.arange(-5, 15, 2), 1)  
C_lbd = [0.03125, 0.125, 0.5, 1.0, 2.0, 8.0, 32.0, 128.0, 512.0, 2048.0, 8192.0]
#C_flash = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
C_flash = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
C_big = [0.0001, 0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,4,8,16,32,64,128,256,512,1024]
gamma_range = numpy.logspace(-9, 3, 13)
gamma_flash=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
gamma_big = [0.0001, 0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2,4,8,16,32,64,100,1000, 3.15,'auto']
# init classifier
svm_init_lbd = {     #'decision_function_shape': None, 
                    'cache_size': 25000, 'gamma': 'auto',
                    'class_weight': None, 'random_state': 42}
metrics = ['accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall']



"""Requeriments: python3.6 -m pip install heapq_max pandas numpy sklearn"""

def selecionar_ids_arquivo(file_ids_total, files_ids_selecionados):
    """
    Objetivo: Selecionar linhas de arquivo a partir de outro arquivo com ids.

    Exemplo: selecionar_ids_arquivo('dataset/produtos_id_titulo.csv', 'dataset/ids_teste')
    """
    dict_id_selecionados = {}
    with open(files_ids_selecionados, newline='') as csvfile_reader_ids_selecionados:
        with open(file_ids_total, newline='') as csvfile_reader_total:
            with open('arquivo_reduzido.csv', 'w', newline='') as csvfile_write_saida:
                ids_selecionados = csv.reader(csvfile_reader_ids_selecionados)
                for row in ids_selecionados:
                    dict_id_selecionados.__setitem__(row[0], True)

                saida_reduzida = csv.writer(csvfile_write_saida, quotechar='|')  # O | permite caracter "
                ids_all = csv.reader(csvfile_reader_total)
                for row in ids_all:
                    if (dict_id_selecionados.__contains__(row[0])):
                        text = "\"" + str(row[1]) + "\""
                        saida_reduzida.writerow([row[0], text])


def arquivo_para_corpus(file_corpus, id_column):
    """ Transforma o arquivo .csv no formato 'list' compatível para o tfidf."""
    corpus = []
    with open(file_corpus, 'r', newline='') as out:
        csv_reader = csv.reader(out, delimiter=',')
        # next(csv_reader)  # Pular cabecalho
        for row in csv_reader:
            corpus.append(row[id_column])
    return corpus

def arquivo_para_corpus_separate_vir(file_corpus, id_column=int):
    """ Transforma o arquivo .csv no formato 'list' compatível para o tfidf."""
    corpus = []
    with open(file_corpus, 'r', newline='') as out:
        csv_reader = csv.reader(out, delimiter=',')
        # next(csv_reader)  # Pular cabecalho
        for row in csv_reader:
            #corpus.append(filter(lambda x: x > id_column, row))
            #lambda a,
            row = row[1:]
            row = ' '.join(row)
            corpus.append(row)
    return corpus

def arquivo_para_corpus_delimiter(file_corpus, delimiter):
    """ Transforma o arquivo .csv no formato 'list' compatível para o tfidf."""
    corpus = []
    with open(file_corpus, 'r', newline='') as out:
        csv_reader = csv.reader(out, delimiter=delimiter)
        # next(csv_reader)  # Pular cabecalho
        for row in csv_reader:
            corpus.append(row)
    return corpus

def file_to_corpus(name_file):
    """Transforma as linahs de um arquivo em uma lista"""
    rows = []
    #with open(name_file, 'r') as read:
    #, encoding = "ISO-8859-1"
    with io.open(name_file, newline='\n', errors='ignore') as read: #erro ignore caracteres
        for row in read:
            row = row.replace("\n", "")
            rows.append(row)
    return rows


def frequencia_termos(listas):
    """ Retorna um dict ordenado pela freqência dos termos da lista."""
    terms_unicos = dict()
    for lista in listas:
        for termo in lista:
            if (terms_unicos.__contains__(termo) == False):
                terms_unicos.__setitem__(termo, 1)
            else:
                terms_unicos.__setitem__(termo, terms_unicos.get(termo) + 1)
    # Ordena pela frequencia, frequencias iguais utiliza ordem alfabetica
    terms_unicos = sorted(terms_unicos.items(), key=lambda e: (-e[1], e[0]))
    return terms_unicos


def soma_das_frequencia(listas):
    """ Soma as frequencias dos termos de uma lista de listas. """
    termos_unicos = {}
    for lista in listas:
        for termo_freq in lista:
            if termos_unicos.__contains__(termo_freq[0]) == True:
                termos_unicos.__setitem__(termo_freq[0], termo_freq[1] + termos_unicos.__getitem__(
                    termo_freq[0]))  # Valor atual + anterior
            else:
                termos_unicos.__setitem__(termo_freq[0], termo_freq[1])
    return sorted(termos_unicos.items(), key=lambda e: (-e[1], e[0]))


def soma_das_frequencia2(listas):
    """ Soma as frequencias dos termos de uma lista de listas. """
    termos_unicos = {}
    for termo_freq in listas:
        if termo_freq[0] in termos_unicos:
            termos_unicos[termo_freq[0]] += termo_freq[1] # Valor atual + anterior
        else:
            termos_unicos[termo_freq[0]] = termo_freq[1]
    return sorted(termos_unicos.items(), key=lambda e: (-e[1], e[0]))


def space_list(lista):
    """ Coloca espaço entre os elementos da lista."""
    return " ".join(lista)


def indice_maior_element_numpy(vet_numpy):
    """ Return the index of max value element in array numpy. """

    result = numpy.where(vet_numpy == numpy.amax(vet_numpy))
    # print('List of Indices of maximum element :', result[1])
    # print(str(result[1]).replace("[", "").replace("]",""))
    return str(result[1]).replace("[", "").replace("]", "")


def save_model_lda(corpus, file_lda):
    """ Salvar modelo LDA em arquivo."""
    print('LDA')
    vectorizer = TfidfVectorizer()  # Utilizar o metodo tfidf
    lda = LatentDirichletAllocation(n_components=10, random_state=0, n_jobs=-1)  # Utilizar 5 topicos
    ini = timeit.default_timer()
    X = vectorizer.fit_transform(corpus)  # Transformar texto em matrix
    # X_TF = vectorizer.fit_transform(corpus).tocsr()
    # X = sp.csr_matrix( ( np.ones(len(X_TF.data)), X_TF.nonzero() ), shape=X_TF.shape )
    lda.fit(X)  # Treinar com o texto
    print("Train LDA: %f" % (timeit.default_timer() - ini))

    # Save all data necessary for later prediction
    dic = vectorizer.get_feature_names()  # nome das features
    model = (dic, lda.components_, lda.exp_dirichlet_component_, lda.doc_topic_prior_)
    with open(file_lda, 'wb') as fp:
        pickle.dump(model, fp)


def save_one_column_csv(file_csv, id_column):
    """ Save one column csv. """
    with open(name_out(file_csv), 'w', newline='') as csv_write:
        rows_out = csv.writer(csv_write, quotechar='|')  # O | permite caracter "
        with open(file_csv, newline='') as csv_reader:
            rows = csv.reader(csv_reader)
            for row in rows:
                rows_out.writerow([row[id_column]])


def name_out(file_csv):
    """ Return name of out file."""
    name = os.path.basename(file_csv)
    file_name = os.path.splitext(name)[0]
    file_type = os.path.splitext(name)[1]
    file_location = os.path.dirname(file_csv) + "/"
    date = "_" + datetime.now().strftime('%d-%m-%Y.%H-%M-%S')
    return file_location + file_name + date + file_type


def name_file(file_csv):
    """ Return name of out file."""
    name = os.path.basename(file_csv)
    file_name = os.path.splitext(name)[0]    
    return file_name


def no_repeat_id(file_csv):
    """ No repeat id in file csv."""
    file_out = name_out(file_csv)
    dict_row = {}

    with open(file_out, 'w', newline='') as csv_write:
        with open(file_csv, newline='') as csv_reader:
            rows = csv.reader(csv_reader, quotechar='|')
            for row in rows:
                dict_row.__setitem__(row[0], row)
            rows_out = csv.writer(csv_write, quotechar='|')  # O | permit character "
            for id, row in dict_row.items():
                rows_out.writerow(row)

def k_max_index(list, k):
    """ Return index of max values.
    Example:
    r = [0.5, 0.7, 0.3, 0.3, 0.3, 0.4, 0.5]
    print(k_max_index(r, 3))
    """''
    list_m = list.copy()
    max_index = []
    k_max_values = nlargest(k, list_m)
    for k_value in k_max_values:
        index_k = list_m.index(k_value)
        max_index.append(index_k)
        list_m[index_k] = -1
    return max_index

def k_max_index2(array, k):
    """ Return index of max values, more eficient."""
    array = numpy.array(array)
    if len(array) < k:
        k = len(array)
    indexs = numpy.argpartition(array, -k)[-k:]
    return indexs[numpy.argsort(-array[indexs])].tolist()



def k_max_index_matrix(matrix, k):
    """ Return index of max values for input matrix.
    Example:
    r = [0.5, 0.7, 0.3, 0.3, 0.3, 0.4, 0.5]
    print(k_max_index(r, 3))
    """''
    list_m = matrix.copy()
    numbers = []
    keywords_fin = []
    for i in range(len(list_m)):
        numbers.append(list_m[i][1])
    max_index = []
    k_max = nlargest(k, numbers)

    for i in k_max:
        max_index.append(list_m.index(i))
        list_m[list_m.index(i)] = -1
    return max_index


def amount_terms_corpus (X, vec_terms):
    """ Amount of terms in corpus """
    terms_total = {}
    for i in range(X.shape[1]):
        terms_total[vec_terms[i]] = sum(X.getcol(i).toarray())[0]
    return terms_total

def save_dict_file(file, dict):
    """Save dict in file"""
    with open(file, 'w', newline='') as csv_write:
        json.dump(dict, csv_write)

def load_dict_file(file):
    """Load dict in file"""
    with open(file, 'r', newline='') as csv_reader:
        return json.load(csv_reader)

def count_same_in_index(list,list2):
    """ Return the amount of value in index."""
    count=0
    for i in range(len(list)):
        if list[i] != 0 and list2[i] != 0:
            count+=1
    return count

def list_files(dir):
    """Return list of the files in directory."""
    files_name =[]
    for r, d, files_array in os.walk(dir):
        for f in files_array:
            files_name.append(f)
    return files_name

def remove_duplicate(mylist):
    """Remve itens duplicate in list."""
    return list(dict.fromkeys(mylist))

def n_list(list, n):
    """Split list in n lists"""
    return numpy.array_split(list, n)

def time_function(fun, params):
    """Calculate time function"""
    #  time_function(name_function, [params..]
    amount_param = len(list(params))

    ini = timeit.default_timer()
    if (amount_param == 0):
        result = fun()
    if (amount_param == 1):
        result = fun(params[0])
    if(amount_param == 2):
        result = fun(params[0], params[1])
    if (amount_param == 3):
        result = fun(params[0], params[1], params[2])
    if (amount_param == 4):
        result = fun(params[0], params[1], params[2], params[3])
    if (amount_param == 5):
        result = fun(params[0], params[1], params[2], params[3], params[4])

    print("Time execute function: %f" % (timeit.default_timer() - ini))
    return result

def average_csv(file):
    "Return list of average of lines csv"
    sum_vector=[]
    with open(file, 'r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            sum_vector.append(statistics.mean([float(r) for r in row]))
    return sum_vector

def standard_deviation(file):
    "Return list of standard deviation of lines csv"
    standard_deviation_list = []
    with open(file, 'r') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            standard_deviation_list.append(statistics.pstdev([float(r) for r in row]))
    return standard_deviation_list

def matrix_confuion(y_true, y_pred, y_labels, columns_labels):
    "Return confusion matrix with title"
    #print(cv.matrix_confuion(y_test, y_pred, [-1,1 ], ['positive', 'negative']))    
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=y_labels),
        index=columns_labels,
        columns=columns_labels
    )
    return matrix.to_string()
    #matrix

def average_csv_column(file, column):
    """Return mean columns specific csv."""
    values=[]
    with open(file, 'r') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        for row in rows:
            values.append(float(row[column]))
        return statistics.mean(values)			

def standard_deviation_column(file, column):
    """Return list of standard deviation of lines csv, specific column."""
    values = []
    with open(file, 'r') as csvfile:
        rows = csv.reader(csvfile)
        next(rows)
        for row in rows:
            values.append(float(row[column]))
    return statistics.pstdev(values)

def flatten(matrix):
    """Return list in 1 dimension"""
    all_folds_test=[]
    for fold in matrix:
        for value in fold:
            all_folds_test.append(value)
    return all_folds_test

def f1(y_test_folds, y_pred_folds, average):
    """Return f1 score of the various lists."""
    metric=[]
    for index in range(len(y_test_folds)):    
        metric.append(sklearn.metrics.f1_score(y_test_folds[index], y_pred_folds[index], average=average))
    return metric

def matrix_confusion_folds(y_test_folds, y_pred_folds, y_labels, columns_labels):
    """Print matrix confunsion per fold."""
    out = ""
    for index in range(len(y_test_folds)):
        out = str_append(out, f"FOLD {index}:")
        out = str_append(out, matrix_confuion(y_test_folds[index], y_pred_folds[index], y_labels, columns_labels))

        #print(f"FOLD {index}:") 
        #print(matrix_confuion(y_test_folds[index], y_pred_folds[index], y_labels, columns_labels))
        #print(matrix_confuion(y_test_folds[index], y_pred_folds[index], [-1,1], ['positive', 'negative']))
    return out

def statistics_experiment(name_dataset, classifier,  y_test_folds, y_pred_folds, best_param_folds, time_method, metric):
    """Write file and print statistics experiment."""
    # encoding = "ISO-8859-1"
    with open(name_out('./statistics/'+name_dataset+"_"+classifier), 'w', errors='ignore') as file_write:
        out = ""
        f1_macro = f1(y_test_folds, y_pred_folds, 'macro')
        f1_micro = f1(y_test_folds, y_pred_folds, 'micro')
        out = str_append(out, "Name Dataset: " + str(name_dataset))
        out = str_append(out, "Best_param_folds: " + str(best_param_folds))
        out = str_append(out, "Macro F1 por fold: " + str(f1_macro))
        out = str_append(out, "Micro F1 por fold: " + str(f1_micro))
        out = str_append(out, 'Média Macro F1: ' + str(statistics.mean(f1_macro)))
        out = str_append(out, "Desvio padrão Macro F1: " + str(statistics.pstdev(f1_macro)))
        out = str_append(out, 'Média Micro F1:  ' + str(statistics.mean(f1_micro)))
        out = str_append(out, "Desvio padrão Micro F1: " + str(statistics.pstdev(f1_micro)))  
        out = str_append(out, "Time method: " + str(max(time_method)))
        classifier = classifier.replace("_", "\\_")

        out = str_append(out, matrix_confusion_folds(y_test_folds, y_pred_folds, numpy.unique(y_test_folds[0]), numpy.unique(y_test_folds[0])))
        if metric == 'f1_macro':
            #print(graph_latex(f1_macro, max(time_method), classifier)) 
            format_latex = str_latex(name_dataset) +"\\_classifier\\_"+classifier +" & " +str('%.2f'%(statistics.mean(f1_macro)*100)) +" (" +str('%.2f'%(statistics.pstdev(f1_macro)*100)) +") & " +str('%.2f'%(max(time_method)))   +" \\\\"
        else:
            #print(graph_latex(f1_micro, max(time_method), classifier) )
            format_latex = str_latex(name_dataset) +"\\_classifier\\_"+classifier  +" & " +str('%.2f'%(statistics.mean(f1_micro)*100)) +" (" +str('%.2f'%(statistics.pstdev(f1_micro)*100)) +") & " +str('%.2f'%(max(time_method)))   +" \\\\"
 

        out = str_append(out, format_latex)
        file_write.write(out)    
    print(format_latex)

def graph_latex(f1, time, classifier):
    """Return format graph latex for plot"""
    f1 = str('%.2f'%(statistics.mean(f1)*100-statistics.pstdev(f1)*100))
    return "\\addplot coordinates{(" +f1 +", " +str(time) +")}; \\addlegendentry{" +classifier +"}"

def str_latex(string):
    """Replace invalid caracters in latex"""
    return string.replace("_", "\\_")   

def save_dict_list(y_true_folds, y_pred_folds, filename):
    """Save predict and true in file"""
    y_pred = dict()
    y_true = dict()
    y=dict()
    for index in range(len(y_pred_folds)):
        y_pred[index]=y_pred_folds[index].tolist()
        y_true[index]=y_true_folds[index].tolist()
    y['y_pred-folds'] = y_pred
    y['y_true-folds'] = y_true
    save_dict_file(filename, y)

def save_dict_list2(y_true_folds, y_pred_folds, filename):
    """Save predict and true in file, no list y_pred"""
    y_pred = dict()
    y_true = dict()
    y=dict()
    for index in range(len(y_pred_folds)):
        y_pred[index]=y_pred_folds[index]
        y_true[index]=y_true_folds[index].tolist()
    y['y_pred-folds'] = y_pred
    y['y_true-folds'] = y_true
    save_dict_file(filename, y)

def rank_grid(rank, k_max):
    """Rank considering standard deviation """  
    rank_scores=[]  
    for itens in rank:
        #rank_scores.append(itens['score']-itens['std'])  
        rank_scores.append(itens['score'])#-itens['std'])  
                
    return k_max_index(rank_scores,k_max)  

def best_param_folds_no_frequency(escores, index_fold, metric):
    """Return best params all folds"""
    index_best = rank_grid(escores[index_fold][metric],1)[0]
    print(escores[index_fold][metric][index_best])
    return escores[index_fold][metric][index_best]['params']

def max_frequency_dict(mydict, field):
    """Return param with maior frequency"""
    frequency=dict()
    for params in mydict:    
        if frequency.__contains__(str(params[field])) == True:
            frequency[str(params[field])]+=1 
        else:
            frequency[str(params[field])] = 1
    max=0  
    print('frequency', frequency)  
    for key, value in frequency.items():
        if value > max:
            max = value
            best_param = key 
            
    #empata
    '''minimun=[]
    for key, value in frequency.items():
        if max == value:
            minimun.append(key)
    if len(minimun)==1:
        return best_param
    else:
        for param in minimun:
            print(param)'''

    return best_param
            
    
def best_param_rank(ranks):  
    """Return best param in rank"""
    params=dict()        
    for index_fold in range(len(ranks)):
        for posicao_rank in range(len(ranks[index_fold])):   
            if params.get(str(ranks[index_fold][posicao_rank]['params'])) != None:
                params[str(ranks[index_fold][posicao_rank]['params'])] = params[str(ranks[index_fold][posicao_rank]['params'])] + posicao_rank
            else:
                params[str(ranks[index_fold][posicao_rank]['params'])] = posicao_rank
    min=100000
    for key, value in params.items():
        if value < min:
            min = value
            best_param = key 
    
    return best_param
    

def best_param_folds(trains, parameter_init, tuned_parameters, metric):
    """Return best params in folds"""
    k=3 # k best params
    top_k_frequency=[]
    
    rank_folds=[]
    for name_train in trains:  
        rank=[]  
        top_k =[]
        x_train, y_train = load_svmlight_files([open(name_train, 'rb')])
        #load_svmlight_file
        grid = GridSearchCV(svm.SVC(**parameter_init), param_grid=tuned_parameters,  cv=5, scoring=metric, n_jobs=1)
        grid.fit(x_train, y_train)   
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            rank.append({'score': mean, 'std': std, 'params': params})                      
        
        for index in rank_grid(rank, k):
            top_k.append(rank[index])
            top_k_frequency.append(rank[index])
            
        rank_folds.append(top_k)
    
    #best_param = best_param_rank(rank_folds)        
    best_param = max_frequency_dict(top_k_frequency, 'params')
    return ast.literal_eval(best_param)

def find_best_param_simple(name_dataset, name_train, estimator, tuned_parameters, metric): 
    """Return best params churando o inicio"""
    atual=0.01       
    top=top_params=ruim=0                
    x_train, y_train = load_svmlight_files([open(name_train, 'rb')])    
    
    while True:
        tuned_parameters = {'kernel' : ['linear'], 'C': [atual]}
        grid = GridSearchCV(estimator, param_grid=tuned_parameters,  cv=5, scoring=metric, n_jobs=1)
        grid.fit(x_train, y_train)  
                       
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            if (mean-std) > top:
                top = (mean-std)
                top_params = params                              
                atual = atual * 2
                ruim=0
            else:     
                atual  = (atual/2) * 1.5                  
                ruim +=1
        if ruim ==2:
            break
    return top_params

def find_best_param(name_dataset, name_train, estimator, tuned_parameters, metric):  
    """Return best params churando o inicio"""      
    atual=1
    anterior=1    
    top=0
    top_params=0
    ruim = 0
    min=0
    max=2
    chute_max=0
    
    x_train, y_train = load_svmlight_files([open(name_train, 'rb')])
    
    
    while True:
        if ruim==0:
            tuned_parameters = {'kernel' : ['linear'], 'C': [atual]}
        else:
            #atual  = (atual/2) + (atual/2) *0.5
            #atual  = top_params['C'] + top_params['C']*(ruim*0.5)
            tuned_parameters = {'kernel' : ['linear'], 'C': [atual]}
    
        grid = GridSearchCV(estimator, param_grid=tuned_parameters,  cv=5, scoring=metric, n_jobs=1)
        grid.fit(x_train, y_train)  
                       
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            if (mean-std) > top:
                top = (mean-std)
                top_params = params
                anterior = atual
                atual = atual*max
                ruim=0
            else:
                chute_max+=1
                max = max/2
                atual = top_params['C'] + (top_params['C']*max)
                ruim +=1
        if ruim ==5:
            break
    ruim=0 
    max=1
    while True:
        if ruim==0:
            tuned_parameters = {'kernel' : ['linear'], 'C': [atual]}
        else:
            #atual  = (atual/2) + (atual/2) *0.5
            #atual  = top_params['C'] + top_params['C']*(ruim*0.5)
            tuned_parameters = {'kernel' : ['linear'], 'C': [atual]}
    
        grid = GridSearchCV(estimator, param_grid=tuned_parameters,  cv=5, scoring=metric, n_jobs=1)
        grid.fit(x_train, y_train)  
                       
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            if (mean-std) > top:
                top = (mean-std)
                top_params = params
                anterior = atual
                max = max/2
                atual = top_params['C'] - (top_params['C']*max)
                ruim=0
            else:
                chute_max+=1
                max = max/2
                atual = top_params['C'] - (top_params['C']*max)
                ruim +=1
        if ruim ==5:
            break                               
    return top_params

def escores_grid(name_dataset, classifier, name_train, estimator, tuned_parameters, metrics, refit):
    """ Return escores grid."""    
    escores=dict()    
    x_train, y_train = load_svmlight_files([open(name_train, 'rb')])
    grid = GridSearchCV(estimator, param_grid=tuned_parameters,  cv=10, scoring=metrics, n_jobs=-1, refit=refit, verbose=30)
    ini = timeit.default_timer()
    grid.fit(x_train, y_train)  
    escores['time_grid'] = timeit.default_timer() - ini
    for metric_value in metrics:
        escores_list=[]
        means = grid.cv_results_['mean_test_'+metric_value]
        stds = grid.cv_results_['std_test_'+metric_value]
        #means = grid.cv_results_['mean_test_score']                
        #stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            escores_list.append({'score': mean, 'std': std, 'params': params})        
        escores[metric_value] = escores_list
    
    file = "escores/"+name_dataset +"_"+classifier +"_escore_grid_"+name_file(name_train)
    try:
        with open(file, 'r') as file_reader: # append params in dict
            old_escore = json.load(file_reader)
            
            for metric_value in metrics:
                if old_escore.get(metric_value) == None:
                    old_escore[metric_value] = escores[metric_value]
                    escores = old_escore
                else:                    
                    escore_list_old = old_escore[metric_value]
                    for params in escores[metric_value]: # add new escores in old escores
                        escore_list_old.append(params)
                    escores[metric_value] = escore_list_old                
    except IOError: 
        pass              

    import claudio_funcoes as cv
    print(cv.best_param_folds_no_frequency([escores], 0, 'f1_macro'))
    save_dict_file(file, escores)
    #joblib.dump(grid, "escores/" + 'grid_'+name_dataset)
    #json.dumps(grid.cv_results_, default=default)

      
    #save_dict_file("escores/"+name_dataset +"_svm_escore_grid_"+name_file(name_train), grid.cv_results_)  

    #return grid

def load_escores(name_dataset, classifier, folds):
    """Load escores"""
    escores=[]
    for index in range(folds):
        escores.append(load_dict_file("escores/"+name_dataset +"_"+classifier +"_escore_grid_train"+str(index)))
    return escores

def best_escore(escores):
    """Return the k best scores"""
    top_k_frequency=[]
    k=3 # k best params
    for fold in escores:
        for index in rank_grid(fold['escores'], 3):
            top_k_frequency.append(fold['escores'][index])            
    best_param = max_frequency_dict(top_k_frequency, 'params')
    return ast.literal_eval(best_param)

def str_append(string, add):
    """Add str in end string"""
    string += str(add) + "\n"    
    return string

def save_list(the_list, filename):
    """Save list in file"""
    with open(filename, 'w') as file_handler:
        for item in the_list:
            file_handler.write("{}\n".format(item))

def ids_train_test(ids_file, datas, labels, id_fold):
    """Return data and labels starting of ids file"""
    ids = file_to_corpus(ids_file)
    train_test = str(ids[id_fold]).split(';')
    ids_train = [int(id) for id in train_test[0].strip().split(' ')]
    ids_test = [int(id) for id in train_test[1].strip().split(' ')]
    total = file_to_corpus(datas)
    labels = file_to_corpus(labels)
    x_train = [total[index] for index in ids_train]       
    y_train = [labels[index] for index in ids_train] 
    x_test = [total[index] for index in ids_test]
    y_test = [labels[index] for index in ids_test] 
    return x_train, y_train, x_test, y_test

def preprocessor(text):
    """ Preprocessoing data. Example: cv.preprocessor('a155a aa10:20b 45638-000')"""        
    replace_patterns = [
    ('<[^>]*>', 'parsedhtml') # remove HTML tags       
    ,(r'(\D)\d\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d:\d\d(\D)', '\\1 ParsedTime \\2') # text_time_text
    ,(r'(\D)\d\d\d\-\d\d\d\d(\D)', 'ParsedPhoneNum') # text_phone_text
    ,(r'(\D)\d\d\d\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2') # text_phone_text
    ,(r'(\D\D)\d\d\d\D\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2') # text_phone_text
    ,(r'(\D)\d\d\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedZipcodePlusFour \\2') # text_zip_text
    ,(r'(\D)\d\d\d\d-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2') # text_phone_text  
    ,(r'\d\d:\d\d:\d\d', 'ParsedTime') #time
    ,(r'\d:\d\d:\d\d', 'ParsedTime') #time
    ,(r'\d\d:\d\d', 'ParsedTime') # time
    ,(r'\d\d\d-\d\d\d\d', 'ParsedPhoneNum') # phone US
    ,(r'\d\d\d\d-\d\d\d\d', 'ParsedPhoneNum') # phone brasil
    ,(r'\d\d\d\d\d\-\d\d\d\d', 'ParsedZipcodePlusFour') # zip
    ,(r'\d\d\d\d\d\-\d\d\d', 'ParsedZipcodePlusFour') # zip brasil
    ,(r'(\D)\d+(\D)', '\\1 ParsedDigits \\2') # text_digit_text
    ,(r'<3', ' good ')
    ,(r'\d+', ' ParsedDigits ') # digit anywhere in the text
    ,(r':\)', ' good ')  # :) :D ;) :( ;( >) ;D
    ,(r':D', ' good ')
    ,(r'=\)', ' good ')
    ,(r'=\]', ' good ')
    ,(r';\)', ' good ')
    ,(r':\(', ' bad ')
    ,(r';\(', ' bad ')
    ,(r'>\)', ' good ')
    ,(r';D', ' good ')
    ,(r':/', ' bad ')
    ,(r'>\(', ' bad ')
    ,(r'>-\(', ' bad ')
    ,(r'>-\)', ' good ')
    ,(r'=/', ' bad ')
    ,(r':-\(', ' bad ')
    ,(r':-\)', ' good ')
    ,(r':]', ' good ')
    ,(r'\(:', ' good ')
    ,(r'\):', ' bad ')
    ,(r'=\(', ' bad ')    
    #,(r'et al', ' autores ')
    ]    
    compiled_replace_patterns = [(re.compile(p[0]), p[1]) for p in replace_patterns]
    
    # For each pattern, replace it with the appropriate string
    for pattern, replace in compiled_replace_patterns:        
        text = re.sub(pattern, replace, text)
    
    #text = remove_caracters_especiais_por_espaco(text)
    text = remove_accents(text)
    text = text.lower()        
    text = remove_point_virgula(text)
    
    text = text.split(" ")
    index=0
    for t in text:
        if text[index].__contains__("http://"):
            text[index] = 'parsedhttp'
        elif text[index].__contains__("@"):
            text[index] = 'parsedref'
        index+=1
    return " ".join(text)

def filter_stop_word(lista, language):
    """Filter stop word in list."""
    stop_words = set(stopwords.words(language))
    filtered_sentence = []
    for w in lista: 
        if w not in stop_words: 
            filtered_sentence.append(w)    
    return filtered_sentence


def my_stop_word(text, language):
    """ Stop word in text"""
    #import nltk
    #nltk.download('punkt')
    #stemmer = SnowballStemmer("english") #stemmer  
    for index in range(len(text)):       
        stop_words = set(stopwords.words(language))         
        #stop_words = set(stopwords.words('portuguese')) 
        word_tokens = text[index].split(" ")
        #word_tokens = word_tokenize(text[index]) #melhor resulto                
        
        filtered_sentence = [] 
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w)
        #filtered_sentence = [stemmer.stem(t) for t in filtered_sentence] #stemmer
        text[index] = " ".join(filtered_sentence)
    return text 

def stemmer_term(term):
    """Stemmer term"""
    stemmer = SnowballStemmer("english") 
    return stemmer.stem(term)

def stemmer_text(text):
    """Stemmer text"""    
    stemmer = SnowballStemmer("english")    
    for index in range(len(text)):        
        stem_list= [stemmer.stem(t) for t in text[index].split(" ")]
        text[index] = " ".join(stem_list)
    return text    

def best_param_dict(mydict, param, metric): 
    """Return best param the dict"""       
    newlist=[]
    for field in mydict[metric]:        
        if field['params'].get(param) != None:
            newlist.append(field)

    best_index = rank_grid(newlist,1)[0]
    return newlist[best_index]    


def create_wordmap(file_text):
    """Return wordmap and frequency terms, format dict"""
    terms = dict()
    set_stop = set(stopwords.words('english')) 
    with open(file_text, 'r') as file_text:
        for line in file_text:            
            line = line.strip()
            row_terms = line.split(' ')
            #terms[row_terms[0]] =1 # o termo apenas no primeiro campo
            for term in row_terms:    
                term = term.lower()
                if term == '' or term in set_stop:
                    continue
                if terms.get(term) == None:
                    terms[term] = 1
                else:
                    terms[term] = terms[term] + 1
    return terms

def join_dict(dict_copy, dict_final):
    """Join two dict, se tem o mesmo id, entao e substituido."""
    for key, value in dict_copy.items():
        dict_final[key] = value
    return dict_final

def join_k_dict(file_location, name_embedding, k):
    """
    Join k dicts.
    Example
    filename = "terms/terms_test0_doc"
    name_embedding = "_wiki-news-300d-1M.vec"
    save_dict_file("join_"+name_file(filename), join_k_dict(filename, name_embedding, 80)) 
    """
    dict_final = dict()
    for index in range(k):    
        if index < 10:
            file = file_location+"_0" + str(index) +name_embedding
        else:
            file = file_location+"_" + str(index) +name_embedding
        dict_final = join_dict(load_dict_file(file), dict_final)
    return dict_final

def join_k_dict_index(file_location, k):
    """ Join k dicts. """
    dict_final = dict()
    for index in range(k):  
        file = file_location + str(index)
        dict_final = join_dict(load_dict_file(file), dict_final)
    return dict_final

def join_dict_docs(index_doc, dict_copy, dict_final):    
    """Join dict doct with index"""
    for key, value in dict_copy.items(): # quantidades documentos                
        dict_final[index_doc] = value
        index_doc+=1
    return dict_final, index_doc

def join_k_dict_docs(file_location, name_embedding, k):
    """
    Join k dicts with index.
    Example
    filename = "terms/terms_test0_doc"
    name_embedding = "_wiki-news-300d-1M.vec"
    save_dict_file("join_"+name_file(filename), join_k_dict(filename, name_embedding, 80)) 
    """
    dict_final = dict()
    index_doc = 0
    for index in range(k): # quantidade de documentos   
        if index < 10:
            file = file_location+"_0" + str(index) +name_embedding
        else:
            file = file_location+"_" + str(index) +name_embedding
        dict_final, index_doc = join_dict_docs(index_doc, load_dict_file(file), dict_final)
    return dict_final



def count_list(values, interval):   
    """Count values in interval""" 
    #interval = [0, 0.09, 0.10, 0.102, 0.105, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 2]        
    #interval = [0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.5]        
    
    count_interval = dict()
    for index in range(len(interval)): # inicializa with zero        
        count_interval[interval[index]] = 0
    for val in values: # cont values inteval        
        val = float(val)
        for index in range(len(interval)-1):
            if val == 0:
                count_interval[interval[index]] = count_interval[interval[index]] + 1
            if val >= interval[index] and val < interval[index+1]:
                count_interval[interval[index]] = count_interval[interval[index]] + 1                                        
    return count_interval
             
def plot_bar_chart(values_dict, title):
    """Plot values"""
    plt.clf()
    y_axis = list(values_dict.values())
    x_axis = list(values_dict.keys())
    #width_n = 0.01    

    #plt.bar(x_axis, y_axis, width=width_n)
    plt.title(title)
    plt.plot(x_axis, y_axis)
    plt.legend()
    plt.xlabel('Score')
    plt.ylabel('Amount')
    #plt.plot([0.1,0.3,0.5,0.8],values_dict)
    #plt.show()    
    plt.savefig(title+'.eps', format='eps')         

def count_interval(matrix):
    """Count values with interval in a matrix"""
    count_interval = dict()
    for index in range(11): # inicializa with zero
        count_interval[index] = 0
        
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if  matrix[row,col] == 0:
                count_interval[0] = count_interval[0] + 1
            elif  matrix[row,col] > 0 and matrix[row,col] <= 0.1:
                count_interval[1] = count_interval[1] + 1    
            elif matrix[row,col] > 0.1 and matrix[row,col] <= 0.2:
                count_interval[2] = count_interval[2] + 1    
            elif matrix[row,col] > 0.2 and matrix[row,col] <= 0.3:
                count_interval[3] = count_interval[3] + 1    
            elif matrix[row,col] > 0.3 and matrix[row,col] <= 0.4:
                count_interval[4] = count_interval[4] + 1    
            elif matrix[row,col] > 0.4 and matrix[row,col] <= 0.5:
                count_interval[5] = count_interval[5] + 1    
            elif matrix[row,col] > 0.5 and matrix[row,col] <= 0.6:
                count_interval[6] = count_interval[6] + 1    
            elif matrix[row,col] > 0.6 and matrix[row,col] <= 0.7:
                count_interval[7] = count_interval[7] + 1    
            elif matrix[row,col] > 0.7 and matrix[row,col] <= 0.8:
                count_interval[8] = count_interval[8] + 1    
            elif matrix[row,col] > 0.8 and matrix[row,col] <= 0.9:
                count_interval[9] = count_interval[9] + 1    
            elif matrix[row,col] > 0.9:
                count_interval[10] = count_interval[10] + 1    
    return count_interval

def same_term(term, terms_expand):
    """Verify if same terms"""
    if stemmer_text(term) == stemmer_text(terms_expand) or term[0]+"s" == terms_expand[0] or term[0] == terms_expand[0]+"s":# or term[0]+"i" == terms_expand[0]:
        return True
    return False

def same_term_list(lista, term_expand):
    """Verify if same terms in lista"""    
        
    for term in lista:        
        if same_term_jaro(term, term_expand):
            return True
        #if stemmer_term(term) == stemmer_term(term_expand) or term+"s" == term_expand or term == term_expand+"s":
        #    return True        
    return False

def same_term_jaro(term, term2):    
    if pyjarowinkler.distance.get_jaro_distance(term, term2, winkler=True, scaling=0.1) >= 0.8:
        return True
    return False
        



def conectar_postgresql():
    """Connect in postgresql"""
    file_secret = load_dict_file("../secret_imoveljson")    
    return psycopg2.connect(host=file_secret['host'], database = file_secret['database_name'], user=file_secret['user'], password=file_secret['password'])    

def execute_postgresql(conection, sql):
    """Execute sql in postgresql"""
    cur = conection.cursor()
    cur.execute(sql)
    cur.close() # close communication with the PostgreSQL database server
    conection.commit() #executa
    
def select_postgresql(conection, sql):
    """Execute select in return result"""
    cur = conection.cursor()
    cur.execute(sql)
    row = cur.fetchone()
    result = []
    while row is not None:   
        result.append(row)         
        row = cur.fetchone()
    cur.close()   
    return result

def select_postgresql_dict(conection, sql):
    """Execute select in return result, result in dict"""
    cur = conection.cursor()
    cur.execute(sql)
    row = cur.fetchone()
    result = dict()
    
    index=0
    while row is not None: 
        row_dict = dict()
        index2=0
        for field in row: 
            if field == None:
                field = "Não cadastrado"
            row_dict[index2] = field                          
            index2+=1
        result[index] = row_dict
        row = cur.fetchone()
        index+=1
    cur.close()   
    return result  

def postgresql_fields_null(post):
    """Trata a entrada para o formato aceito no postgresql"""
    newfields = dict()
    for field in post:    
        if post.get(field) == None:
            newfields[field] = "null"
        else:
            newfields[field] = post[field]                            
            
    return newfields

def fields_null(request):
    """Trata a entrada para o formato aceito no postgresql"""
    newfields = dict()
    for field in request.POST:    
        if request.POST[field] == "":
            newfields[field] = "null"
        else:
            newfields[field] = '\'' + request.POST[field] +'\''
    return newfields


def send_email(to_addrs, subject, message, password):
    # conexão com os servidores do google
    smtp_ssl_host = 'smtp.gmail.com'
    smtp_ssl_port = 465
    # username ou email para logar no servidor
    username = 'claudiovaliense@gmail.com'

    from_addr = 'claudiovaliense@gmail.com'    

    # a biblioteca email possuí vários templates
    # para diferentes formatos de mensagem
    # neste caso usaremos MIMEText para enviar
    # somente texto
    message = MIMEText(message)
    message['subject'] = subject
    message['from'] = from_addr
    message['to'] = ', '.join(to_addrs)

    # conectaremos de forma segura usando SSL
    server = smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port)
    # para interagir com um servidor externo precisaremos
    # fazer login nele
    server.login(username, password)
    server.sendmail(from_addr, to_addrs, message.as_string())
    server.quit()


def arredonda(number):
    return str('%.2f'%(number))

def arredondar_numero(list_list):            
    for list in list_list:         
        for index in range(len(list)):
            if type(list[index]) == float:
                list[index] = str('%.2f'%(list[index])) # arredonda indice especifico
    return list_list

def reduce_dict(mydict, k):
    """Get k first in dict"""
    newdict = dict() 
    index = 0
    for key, value in mydict.items():
        if index < k:
            newdict[key] = value
        else:
            break
        index+=1
    return newdict

def remove_none(mydict):
    """Remove value none in dict"""
    temp = []
    for key, value in mydict.items(): # remove actions sell
        if value == None:
            temp.append(key)
    for key in temp:
        del mydict[key]
    return mydict

def remove_none_list(array):
    """Remove None in list"""
    new_array = []
    for index in range(len(array)):
        if array[index] != None:
            new_array.append(array[index])
    return new_array

def remove_none_numpy(array):
    """Remove None in numpy"""
    new_array = []
    for index in range(len(array)):
        if numpy.isnan(array[index][0]) == False:
            new_array.append(array[index])
    return new_array

def specific_index(array, specific):
    """Return new array with list the specific index"""
    new_array = []
    for index in range(len(array)):
        if index in specific:
            new_array.append(array[index])
    return new_array

def not_specific_index(array, specific):
    """Return new array no list specific index"""
    new_array = []
    for index in range(len(array)):
        if index not in specific:
            new_array.append(array[index])
    return new_array

def set_none(array, specific_index):   
    """Set None in list of specific index""" 
    for index in range(len(array)):
        if index in specific_index:
            array[index] = None
    return array

def balance_class(x_train, y_train):
    """Balance class, two class"""
    index_class[0]  = []
    index_class[1]  = []
    for index in range(len(y_train)):
        if y_train[index] == 1:
            index_class[1].append(index)            
        else:
            index_class[0].append(index)
    
    print('cont0:', len(index_class[0]))
    print('cont1:', len(index_class[1])) 
    dif = abs(len(index_class[0]) - len(index_class[1]))
    if len(index_class[0]) > len(index_class[1]):
        x_train = not_specific_index(x_train, index_class[0][0:dif])        
        y_train = not_specific_index(y_train, index_class[0][0:dif])
    elif len(index_class[1]) > len(index_class[0]):
        x_train = not_specific_index(x_train, index_class[1][0:dif])        
        y_train = not_specific_index(y_train, index_class[1][0:dif])

def get_wordmap(filename):
    """Get wordmap in format: term id"""
    wordmap_file = open(filename, 'r')
    wordmap = dict()

    for row_word in wordmap_file: # colocando o wordmap em um dict                
        row_word = row_word.strip().split(" ")               
        wordmap[row_word[1]] = row_word[0] # format:: term id
    return wordmap

def remove_accents(string):
    """ Remove accents string. """
    return unidecode.unidecode(string)

def value_to_key(mydict):
    """ Swap key to value."""
    newdict = dict()
    for k, v in mydict.items():
        newdict[v]= k
    return newdict

def add_column_csv(filename, lista, name_field):
    """ Add column in end csv."""
    file_r = open(filename, 'r')
    writer = open(filename+"_"+name_field, "w")
    index=0
    for line in file_r:
        line = line.strip() +"," +"\"" +lista[index] +"\"\n"         
        writer.write(line)
        index+=1
    writer.close()

def url_encode(string):
    """ Transform string in format accept per url."""
    return urllib.parse.quote(string)

def add_field_json(mydict, name_field, new_field_list):
    """ Adicione new field in dict."""
    index=0
    for key in mydict.keys():
        mydict[key].update({name_field : new_field_list[index]})
        index+=1
    return mydict

def k_dict(mydict, k):
    """ Return newdict with k keys."""
    newdict = dict()
    cont=1
    for key, value in mydict.items():
        if cont > k: 
            break
        newdict[key] = value
        cont+=1
    return newdict

def remove_key_dict(mydict, keys):
    """Return new dict sem as keys"""
    new_dict = dict()
    for key in mydict:
        if key not in keys:
            new_dict[key] = mydict[key]
    return new_dict

def remove_key_key_dict(mydict, keys):
    """Return new dict sem as keys"""
    new_dict = dict()
    for key, v in mydict.items():
        for key2 in v:
            if key2 not in keys:
                if new_dict.get(key) == None:
                    new_dict[key] = {}
                new_dict[key][key2] = mydict[key][key2]
    return new_dict

def save_specific_field(mydict, field, filename):
    escreve = open(filename, 'w')
    for key, value in mydict.items():        
        escreve.write(("\"" +key + "\",\"" +value[field] +"\"\n"))

def intervalo_confianca(data):
    return np.mean(data) - stats.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=stats.sem(data))[0]

def disc(k):
    return 1.0 / math.log(k + 2.0, 2.0)

def ndcg(gabarito, res):   
    MAX_REC = 1000
    discounts = [disc(i) for i in range(MAX_REC)]
    idcgs = [discounts[0]]
    for i in range(len(discounts) - 1):
        idcgs.append(idcgs[i] + discounts[i+1])        
    
    if len(res) == 0:
        return 0
    dcg = 0.0
    for i in range(len(res)):
        if res[i] in gabarito:
            dcg += discounts[i]
    m = len(res) - 1
    return dcg / idcgs[m]

def quantify_key_key(mydict, key2):
    """Quantifica uma chave dentro de um dicionario"""
    cont=0
    for key in mydict:
        if mydict[key].get(key2):
            cont+=1
    return cont

def csv_to_dict(filename):
    """ Convert csv in dict."""
    mydict = dict()    
    for row in csv.reader(open(filename)):
        mydict[row[0]] = []
        for index in range(1,len(row)):
            mydict[row[0]].append(row[index])
    return mydict

def filter_nome(terms, termos_objeto):
    """Filter nome proprio que nao esteja na lista"""
    #nltk.download('averaged_perceptron_tagger') # Download do dicionario identificador
    new_list = []
    for term in terms:
        term = term.capitalize()
        tokens = nltk.word_tokenize(term)
        for term_tipo in nltk.pos_tag(tokens):
            if term_tipo[1] == 'NNP' and term_tipo[0] not in termos_objeto:
                print(term_tipo)
                continue
            new_list.append(term_tipo[0])
    return new_list

def list_keys(mydict, keys):
    """Return list contain all data das keys. """
    all = dict()        
    for k, v in mydict.items():
        new_lista = []        
        for k2 in keys:
            for tag in v[k2]:
                new_lista.append(tag[0])
        all[k] = new_lista
    return all

def remove_none_key2(mydict, keys2):
    """Remove value none in dict"""
    temp = []
    for key, value in mydict.items(): # remove actions sell
        for k2 in keys2:
            if value.get(k2) == None or value.get(k2) == '':# or value.get(k2) == '\\N':
                print('Chave ausente:', k2)
                temp.append(key)
                break
    for key in temp:
        del mydict[key]
    return mydict

def total_termos(filename):
    """Return total terms in file"""
    terms = set()
    for line in open(filename, encoding='ISO-8859-1'):
        for word in line.strip().split(" "):
            terms.add(word)
    return terms
    #return len(terms)

def total_termos2(lista_doc):
    """Return total terms in file"""
    terms = set()
    for doc in lista_doc:
        for word in doc.strip().split(" "):
            terms.add(word)
    return terms

def remove_caracters_especiais_por_espaco(text):
    text =  re.sub("[()!;':?><,.?/+-=-_#$%ˆ&*]", " ", text)
    return re.sub(' +', ' ', text) # remove multiple space

def remove_point_virgula(text):
    text =  re.sub("[.,]", " ", text)
    return re.sub(' +', ' ', text) # remove multiple space

def split_dict_k(mydict, k_split, namefile):    
    """Split dict in k files"""
    per_file = math.ceil(len(mydict.keys()) / k_split)
    cont=0 
    index_file=0   
    new_dict = dict()
    for k in mydict:
        if cont == per_file:            
            save_dict_file(namefile+str(index_file), new_dict)
            new_dict = dict()
            cont=0
            index_file+=1
        new_dict[k] = mydict[k]
        cont+=1
    save_dict_file(namefile+str(index_file), new_dict)

def add_key2_in_other_dict(from_dict, to_dict, key2):
    for k in from_dict:
        if from_dict[k].get(key2) != None:
            to_dict[k][key2] = from_dict[k][key2]  
    return to_dict          

def join_text_label(text, label):
    """ Join text and label. 
    Example: join_text_label(stanford_tweets/texts.txt, stanford_tweets/label.txt)"""
    with open(text+"_"+name_file(label)+".csv", 'w') as write:        
        text = open(text, 'r')
        label = open(label, 'r').readlines()        
        for index_label in range(len(label)):
            if index_label+1 != len(label): # no add \n in last line           
                x = "\"" + label[index_label].strip() + "\",\"" +text.readline().strip() +"\"\n"
            else:              
                x = "\"" + label[index_label].strip() + "\",\"" +text.readline().strip() +"\""          
            write.write(x)            

def split_train_test(text_label, percentage):
    """Split train test in percentage"""
    with open(text_label+"_train", 'w') as train:
        with open(text_label+"_test", 'w') as test:  
            text_label = open(text_label, 'r').readlines()                         
            limit_train = int((len(text_label) * percentage) / 100)
            for index in range(len(text_label)):
                if index < limit_train:
                    if index+1 != limit_train: # no add \n in last line
                        train.write(text_label[index])
                    else:
                        train.write(text_label[index].strip())
                else:
                    test.write(text_label[index])    

def plot_histogram(values_dict, title):
    plt.clf()
    plt.title(title)
    plt.hist(list(values_dict.values()), bins=50)
    plt.show()

def token_text(texts_ori):
    #return word_tokenize(texts)
    texts = texts_ori.copy()
    for index in range(len(texts)):
        word_tokens = word_tokenize(texts[index])
        texts[index] = word_tokens
        #texts[index] = " ".join(word_tokens)
    return texts
    #word_tokens = text[index].split(" ")



def clean_text(text):
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words("english")) 
    lemmatizer = WordNetLemmatizer() 
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def format_lbd():
    dataset = 'learning-nlp'
    filename = 'dataset/learning-nlp/train.csv'
    file = csv.reader(open(filename))
    #file = pd.read_csv(filename)
    texts = open('dataset/'+dataset+"/orig/texts.txt", "w")
    score = open('dataset/'+dataset+"/orig/score.txt", "w")
    next(file)
    
    #print('text:', len(file['text']))
    #print('label:', len(file['class_label']))    
    #x = [texts.write(doc +"\n") for doc in file['text']]
    #x = [score.write(str(doc)+"\n") for doc in file['class_label']]
    
    cont=0
    for line in file:    
        texts.write(line[2].replace("\n", " ").strip() +"\n")    
        score.write(line[4] +"\n")        
        cont+=1
    print(cont)
    

def load_test(dataset):
    
    filename = 'dataset/' +dataset +'/test.csv'
    #filename = 'dataset/' +dataset +'/testData.tsv'
    file = csv.reader(open(filename), delimiter="\t")
    next(file)
    x_test = []
    y_test = []
    for line in file:
        x_test.append(line[1])
    return x_test

def load_test2(dataset):    
    filename = 'dataset/' +dataset +'/test.csv'    
    file = csv.reader(open(filename))
    next(file)
    x_test = []        
    for field in file:           
        text_agg=[]
        text_agg.append(field[2])
        text_agg.append(field[4])
        text_agg.append(field[6])
        text_agg = " ".join(text_agg)        
        x_test.append(text_agg.strip())            
    
    return x_test

def docs_kaggle(dataset):    
    file_texts = 'dataset/' +dataset +'/orig/texts.txt'
    file_score = 'dataset/' +dataset +'/orig/score.txt'              
    x_train = [x.strip() for x in open(file_texts)]
    y_train = [x.strip() for x in open(file_score)]
    x_test = load_test2(dataset)    
    y_test = []
    return x_train, y_train, x_test, y_test

def docs_kaggle2(dataset):    
    file_texts = 'dataset/' +dataset +'/orig/texts2.txt'
    file_score = 'dataset/' +dataset +'/orig/score.txt'              
    x_train = [x.strip() for x in open(file_texts)]
    y_train = [x.strip() for x in open(file_score)]
    x_test = load_test2(dataset)    
    y_test = []
    return x_train, y_train, x_test, y_test

def file_submit_kaggle(dataset):
    result = load_dict_file('y_pred/' +dataset +'_passive_aggressive_f1_macro_test0')
    #result = load_dict_file('y_pred/learning-nlp_svm_f1_macro_test0')
    saida = open('dataset/' +dataset +'/kaggle.csv', 'w')
    id = 0
    saida.write("Id,Expected\n")
    for v in result['y_pred-folds']['0']:
        saida.write(str(id)+"," +str(int(v)) +"\n")
        id+=1

def file_submit_kaggle2(dataset):
    result = load_dict_file('y_pred/' +dataset +'_passive_aggressive_f1_macro_test0')
    #result = load_dict_file('y_pred/learning-nlp_svm_f1_macro_test0')
    saida = open('dataset/' +dataset +'/kaggle.csv', 'w')
    
    filename = 'dataset/' +dataset +'/testData.tsv'
    file = csv.reader(open(filename), delimiter="\t")
    next(file) # pula cabecalho
    id = []
    for line in file:
        id.append(line[0])
    
    saida.write("\"id\",\"sentiment\"\n")
    index_id = 0
    for v in result['y_pred-folds']['0']:
        saida.write("\"" +id[index_id] +"\"," +str(int(v)) +"\n")        
        index_id+=1
        
def file_submit_kaggle3(dataset, classificador):
    #result = load_dict_file('y_pred/' +dataset +'_random_forest_f1_macro_test0')    
    result = load_dict_file('y_pred/' +dataset +'_'+classificador +'_f1_macro_test0')    
    #result = load_dict_file('y_pred/' +dataset +'_svm_f1_macro_test0') 
    saida = open('dataset/' +dataset +'/kaggle.csv', 'w')
    
    filename = 'dataset/' +dataset +'/test.csv'
    file = csv.reader(open(filename))
    next(file) # pula cabecalho
    id = []
    for line in file:
        id.append(line[0])
    
    saida.write("unique_id,citation_influence_label\n")
    #saida.write("unique_id,citation_class_label\n")
    index_id = 0
    for v in result['y_pred-folds']['0']:
        saida.write(id[index_id] +"," +str(int(v)) +"\n")        
        index_id+=1

def format_lbd2():
    dataset = 'word2vec-nlp-tutorial'
    filename = 'dataset/' +dataset +'/labeledTrainData.tsv'
    file = csv.reader(open(filename), delimiter="\t")
    #file = pd.read_csv(filename)
    texts = open('dataset/'+dataset+"/orig/texts.txt", "w")
    score = open('dataset/'+dataset+"/orig/score.txt", "w")
    next(file)
    
    #print('text:', len(file['text']))
    #print('label:', len(file['class_label']))    
    #x = [texts.write(doc +"\n") for doc in file['text']]
    #x = [score.write(str(doc)+"\n") for doc in file['class_label']]
    
    cont=0
    for line in file:    
        texts.write(line[2].replace("\n", " ").strip() +"\n")    
        score.write(line[1] +"\n")        
        cont+=1
    print(cont)

def termos_texts(texts):
    """Return list terms per document."""
    text_termos = []    
    for text in texts:
        term_text=[]
        for t in text.strip().split(" "):
            term_text.append(t)
        text_termos.append(term_text)
    return text_termos

def classes_doc(y_train):
    """Return set classes in y_train."""
    classes = set()
    for y in y_train:
        classes.add(y)
    return classes

def doc_specific_class(x_train, y_train):
    """Return x dividido por classes in json"""
    x_specific = dict()
    index=0
    for y in y_train:        
        if x_specific.get(y) == None:
            x_specific[y] = []
        x_specific[y].append(x_train[index])
        index+=1
    return x_specific

def statistics_dataset():
    """Statistics Dataset"""
    name_tipos = [['3c-shared-task-purpose', 'Topic'], ['3c-shared-task-influence', 'Influence'],
                 ['stanford_tweets', 'Sentiment'], ['yelp_review_2L', 'Sentiment'],
                 ['webkb', 'Topic'], ['learning-nlp', 'Sentiment'], ['reut', 'Topic'],
                 ['20ng', 'Topic'], ['acm', 'Topic'], ['word2vec-nlp-tutorial', 'Sentiment'],
                 ['agnews', 'Topic']]
    for name_tipo in name_tipos:
        name = name_tipo[0]
        dataset = 'dataset/'+name
        x = file_to_corpus(dataset+'/orig/texts.txt')
        qtd_docs = len(x)
        #for index in range(len(x)):
            #x[index] = x[index].lower()
            #x[index] = remove_accents(x[index])
            #x[index] = remove_caracters_especiais_por_espaco(x[index])
            
        y = file_to_corpus(dataset+'/orig/score.txt')        
        qtd_terms_unicos = len(total_termos2(x))                
        docs_termos =  termos_texts(x)
        docs_stop = []
        for terms in docs_termos:            
            docs_stop.append(filter_stop_word(terms, 'english'))
        
        docs_stop = [len(terms) for terms in docs_stop]             
        termos_docs = [len(terms) for terms in docs_termos]    
        qtd_class = len(classes_doc(y))              
        class_docs = doc_specific_class(x, y)
        qtd_docs_class = [len(docs) for docs in class_docs.values()]
        
        '''
        print('Quantidade de documentos: ', qtd_docs)
        print('Quantidade de termos únicos: ', qtd_terms_unicos)
        print('Quantidade de total de termos: ', sum(termos_docs))    
        print('Média de termos por documento: ', statistics.mean(termos_docs))
        print('Mediana de termos por documento: ', statistics.median(termos_docs))
        print('Quantidade de classes: ', qtd_class)
        print('Quantidade de documentos da maior classe: ', max(qtd_docs_class))
        print('Quantidade de documentos da menor classe: ', min(qtd_docs_class))
        print('Média do tamanho da classe: ', statistics.mean(qtd_docs_class))
        '''
            
        
        '''latex = "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
            name.replace("_", " ").capitalize(), qtd_docs,  sum(termos_docs), 
            qtd_terms_unicos, arredonda(statistics.mean(termos_docs)),
            arredonda(statistics.median(termos_docs)), qtd_class, max(qtd_docs_class), min(qtd_docs_class),
            arredonda(statistics.mean(qtd_docs_class)), sum(docs_stop), name_tipo[1])    '''
        
        latex = "{} & {} & {} & {} & {} & {}  & {} & {}  \\\\".format(
            name.replace("_", " ").capitalize(), qtd_docs,  sum(termos_docs), 
            qtd_terms_unicos, 
            arredonda(statistics.median(termos_docs)), qtd_class, max(qtd_docs_class), min(qtd_docs_class) )  
        print(latex)

def format_lbd3():
    #dataset = '3c-shared-task-influence'
    dataset = '3c-shared-task-purpose'
    filename = 'dataset/'+dataset +'/train.csv'
    file = csv.reader(open(filename))    
    texts = open('dataset/'+dataset+"/orig/texts.txt", "w")
    score = open('dataset/'+dataset+"/orig/score.txt", "w")
    next(file)    
    
    cont=0
    
    for field in file:           
        text_agg=[]
        text_agg.append(field[2])
        text_agg.append(field[4])
        text_agg.append(field[6])
        #if field[3].__contains__(field[5]):            
        #    text_agg.append("parsedSameAuthor")         
        text_agg = " ".join(text_agg)        
        texts.write(text_agg.strip() +"\n")    
        score.write(field[7] +"\n")        
        cont+=1
    print(cont)

def count_label(data, labels):
    new = pd.DataFrame()
    df = pd.DataFrame(data)    
    df['target'] = labels
    return df.target.value_counts()
    
    '''min_class = target_count.min(axis=0)
    
    for label in list(target_count.keys()):                
        df_class_0 = df[df['target'] == label]    
        df_class_0_under = df_class_0.sample(min_class)        
        df[df['target'] == label] = df_class_0_under
        #new.append(pd.DataFrame(df_class_0_under))   '''     

def undersample(X, y, strategy):
    undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=strategy)
    return undersample.fit_resample(X, y)

def oversample(X, y, strategy):
    # define oversampling strategy
    # sampling_strategy='minority', 'not minority', 'all', 'auto'
    oversample = imblearn.over_sampling.RandomOverSampler(sampling_strategy=strategy)
    return oversample.fit_resample(X, y)

def similarity_jaro(str1, str2):
    #return distance.ja
    return distance.get_jaro_distance(str1, str2)



#print("12 33 22 3322".split(" "))
#statistics_dataset()
#file_submit_kaggle3('3c-shared-task-influence', 'passive_aggressive')
#file_submit_kaggle3('3c-shared-task-purpose', 'passive_aggressive')
#format_lbd3()
#from sklearn.feature_extraction import stop_words
#print(stop_words.ENGLISH_STOP_WORDS)
#format_lbd3()
#format_lbd2()
#file_submit_kaggle2('word2vec-nlp-tutorial')        
#format_lbd()
#docs_kaggle()
#print(add_key2_in_other_dict({'a': {'t': 'casa'}}, {'a' : {'b': 2}}, 't') )
#save_dict_file('movielens_fiji', join_k_dict_index('../tagrec/movielens/parallel/all_data_avaliation.json', 64))

# plot similarity average
'''average = load_dict_file('join_movielens_max_pooling_0.out.test')
vet_sum = []
for key, value in average.items():
    for v in value:
        vet_sum.append(v[2])

plot_bar_chart(count_list(vet_sum, [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]), "similarity_max_pooling")'''

#filename = "../dataset/stanford_tweets/orig/texts.txt_pre"
#save_dict_file(filename+"_wordmap", create_wordmap(filename))


#print(count_list(values))
#plot_bar_chart(count_list(values))
        
#juntar termos varios dicionarios
#filename = "doc_split/0.out.test.rec"
#filename = "doc_split/0.test.ids"
#name_embedding = "_terms_lastfm_wordmap_unstemmed_wiki-news-300d-1M.vec_reduce"
#name_embedding="_movielens_average"
#save_dict_file("join_original_movielens_average_"+name_file(filename), join_k_dict_docs(filename, name_embedding, 65))


#file_text = "lastfm_styles.unstemmed_words"
#file_terms = "terms_lastfm_wordmap_unstemmed"
#save_list(list(text_to_dict_terms(file_text)), file_terms)

#send_email()
#mydict = load_dict_file('escores/stanford_tweets_10_folds_tfidf_claudio_random_forest_escore_grid_train0')    
#a = best_param_dict(mydict, 'max_depth', 'f1_macro')
#print(a)
#print(average_csv_column('/home/claudiovaliense/projetos/metalazy/metalazy/experiments/results/result_tunning_time.csv', 1))
#print(standard_deviation_column('/home/claudiovaliense/projetos/metalazy/metalazy/experiments/results/result_tunning_time.csv', 1))
#print('Average: ',average_csv('f1.csv'))
#print('standard_deviation', standard_deviation('f1.csv'))

"""
Nao sei o que fazer com esses codigos, backup
def default(obj):
    if type(obj).__module__ == numpy.__name__:
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


count_process = 0
def calculate_process(size):
    global count_process
    count_process += 1
    print("Process: ", count_process, "/", size, ", ", (count_process/size)*100," %")    
"""
