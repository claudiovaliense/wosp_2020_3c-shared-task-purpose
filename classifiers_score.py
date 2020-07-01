from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm  # Classifier SVN
from sklearn.naive_bayes import MultinomialNB  # Classifier MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier # Classifier ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
import numpy # Manipulate array
import timeit  # Measure time
import sys # Import other directory
import claudio_funcoes as cv  # Functions utils author
from sklearn.ensemble import RandomForestClassifier

""" Example execution: python3.6 name_dataset file_train file_test"""
SEED=42
numpy.random.seed(seed=SEED)
ini = timeit.default_timer() # Time process

def run_classifier(name_dataset, file_train, classifier):
    """Run classifier"""            
    if classifier == "ada_boost":
        tuned_parameters =[{
            'n_estimators': [100],
            'learning_rate': [0.1]#,
            #'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
            #'algorithm': ['SAMME.R']
        }]
        tuned_parameters = [{}]
        estimator = AdaBoostClassifier(random_state=42, base_estimator= ComplementNB(alpha=0.01))
    
    elif classifier == "extra_tree":
        tuned_parameters =[{
            'n_estimators': [200],
            'max_features':['auto', None],
            'min_samples_split':[2,5,10],
            'min_samples_leaf':[1,5,10]
        }]
        estimator = ExtraTreesClassifier(random_state=SEED)
    
    elif classifier == "knn":
        tuned_parameters =[{
            'n_neighbors': [1, 5, 10, 50, 100, 500, 1000],
            'weights':['uniform', 'distance'],
            'p':[1,2]
        }]
        estimator = KNeighborsClassifier()
    
    elif classifier == "logistic_regression":
        tuned_parameters =[{            
            'C':[0.1],
            #'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
            'solver' : ['sag' ]
            #'max_iter':[500]
        }]
        estimator = LogisticRegression(random_state=SEED)
    
    elif classifier == "naive_bayes":
        tuned_parameters =[{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10] }]
        estimator = MultinomialNB()
    
    elif classifier == "naive_bayes_complement":
        tuned_parameters =[{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10] }]
        estimator = ComplementNB()
    
    elif classifier == "passive_aggressive":
        tuned_parameters =[{
            #'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10, 50, 100],         
            'C': [ 0.0001,  0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]#,
            #'max_iter' : [1000]
            #'loss': ['hinge', 'squared_hinge'],
            #'fit_intercept' : [False, True]            
        }]
        estimator = PassiveAggressiveClassifier(random_state=SEED, max_iter = 1000)
    elif classifier == "random_forest":
        tuned_parameters =[{   
            'n_estimators': [100],
            'min_samples_leaf': [1]
            #'class_weight': ['balanced', None]#,
                    
        }]
        estimator = RandomForestClassifier(random_state=SEED)
        
    elif classifier == "sgd":
        #tuned_parameters =[{'alpha': [0.09419006441400779], 'eta0': [0.2573318908579897], 'learning_rate': ['optimal'], 'loss': ['perceptron']}]
        tuned_parameters = [{'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1], 'max_iter': [1000]}]#,
                             #"loss" : ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]}]
        #tuned_parameters = [{'loss': ['log']}]
        '''tuned_parameters =[{
            'max_iter': [1000],
            'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1],
            "loss" : ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            'penalty':['l1', 'l2'],
            'learning_rate': ['optimal', 'constant', 'invscaling', 'adaptive'],
            "eta0":[0.0001, 0.001, 0.01]
        }]'''
        estimator = SGDClassifier(random_state=SEED, max_iter=1000)

    elif classifier == "svm":        
        #tuned_parameters = [{'kernel': ['rbf'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}]
        tuned_parameters = [{'C': [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000] }]
    
        #estimator = svm.SVC(random_state=SEED)        
        estimator = svm.LinearSVC(random_state=SEED, max_iter=1000)

    elif classifier == "voting":
        tuned_parameters =[{
            'voting': ['hard']#, 'soft']            
        }]
        classifiers = [('naive_bayes', MultinomialNB()), ('naive_bayes_complement', ComplementNB())]
        estimator = VotingClassifier(estimators=classifiers)

    cv.escores_grid(name_dataset, classifier, file_train, estimator, tuned_parameters, ['f1_macro', 'f1_micro'], 'f1_macro') #best param fold

name_dataset = sys.argv[1]
file_train = sys.argv[2]
classifier = sys.argv[3]
run_classifier(name_dataset, file_train, classifier)
print("Time End: %f" % (timeit.default_timer() - ini))
