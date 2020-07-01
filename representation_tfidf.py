import timeit  # Measure time
import sys # Import other directory
import claudio_funcoes as cv  # Functions utils author
from sklearn.feature_extraction.text import TfidfVectorizer # representation tfidf
from sklearn.datasets import dump_svmlight_file # save format svmlight
import os
import nltk
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

def glove_dict(X, x_test):
    import struct 

    glove_small = {}
    #all_words = set(w for words in X for w in words)
    all_words = cv.total_termos2(X)
    all_words_test = cv.total_termos2(x_test)
    for t in all_words_test:
        all_words.add(t)
    
    with open('../glove.6B.300d.txt', "rb") as infile:    
        for line in infile:
            parts = line.split()
            word = parts[0].decode('utf-8')
            if (word in all_words):
                nums=np.array(parts[1:], dtype=np.float32)
                glove_small[word] = nums
    return glove_small

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:            
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):       
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

ini = timeit.default_timer() # Time process
name_dataset = sys.argv[1]
ids=sys.argv[2]
datas=sys.argv[3]
labels=sys.argv[4]
index = int(sys.argv[5])

try:
    os.mkdir("dataset/representations/"+name_dataset) # Create directory
except OSError:
    print('directory exist')

x_train, y_train, x_test, y_test = cv.docs_kaggle(name_dataset) #kaggle


y_test = [1 for x in range(len(x_test))]   #kaggle
x_train = [cv.preprocessor(x) for x in x_train]
x_test = [cv.preprocessor(x) for x in x_test]
glove_small = glove_dict(x_train, x_test)

y_train = [float(y) for y in y_train] # float para permitir utilizar no classificador
y_test = [float(y) for y in y_test]   

union = Pipeline([                         
                       ('features',   FeatureUnion(transformer_list=[
                            
                           ('tfdif_features', Pipeline([                                
                                ('word', TfidfVectorizer(ngram_range=(1,2)) )#,                      
                           ])),
                           ('topico_nmf', Pipeline([
                                ('word', TfidfVectorizer(ngram_range=(1,2)) ),                                                                  
                                ('lda', LatentDirichletAllocation(n_components=6, random_state=42) )                                                            
                                                            
                           ])),                                                  
                           ('pos_features', Pipeline([                                                                                                                                            
                               ('mean_word2vec', MeanEmbeddingVectorizer(glove_small))
                           ]))
                        ]
                        
                        ))
])

x_train = union.fit_transform(x_train)
print('fold ' +str(index) +', x_train.shape: ', x_train.shape)    
dump_svmlight_file(x_train, y_train, "dataset/representations/" + name_dataset +'/train'+str(index))

x_test = union.transform(x_test)
print('fold ' +str(index) +', x_test.shape: ', x_test.shape)
dump_svmlight_file(x_test, y_test, "dataset/representations/" + name_dataset +'/test'+str(index))
print("Time End: %f" % (timeit.default_timer() - ini))
