from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm  # Classifier SVN
from sklearn.naive_bayes import MultinomialNB  # Classifier MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier # Classifier ExtraTreesClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


from sklearn.datasets import load_svmlight_file, load_svmlight_files # import file svmlight
import numpy # Manipulate array
import timeit  # Measure time
import sys # Import other directory
import claudio_funcoes as cv  # Functions utils author
import joblib

""" Example execution: python3.6 svm_predict.py name_dataset file_train file_test num_folds"""

def predict_classifier(name_dataset, name_train, classifier, name_test, metric):
    """Run classifier"""            
    if classifier == "ada_boost":        
        estimator = AdaBoostClassifier(random_state=42, base_estimator= ComplementNB(alpha=0.01))
        #estimator = AdaBoostClassifier(random_state=42, base_estimator= LogisticRegression(C= 50, max_iter= 100))
    
    elif classifier == "extra_tree":        
        estimator = ExtraTreesClassifier(random_state=SEED)
    
    elif classifier == "knn":        
        estimator = KNeighborsClassifier()
    
    elif classifier == "logistic_regression":        
        estimator = LogisticRegression(random_state=SEED)
    
    elif classifier == "naive_bayes":        
        estimator = MultinomialNB()
    
    elif classifier == "naive_bayes_complement":        
        estimator = ComplementNB()
    
    elif classifier == "passive_aggressive":        
        estimator = PassiveAggressiveClassifier(random_state=SEED, max_iter=1000)
    
    elif classifier == "random_forest":        
        estimator = RandomForestClassifier(random_state=SEED)        
        
    elif classifier == "sgd":        
        estimator = SGDClassifier(random_state=SEED, max_iter=1000)

    elif classifier == "svm":                
        estimator = svm.LinearSVC(random_state=SEED, max_iter=1000)        
    
    x_train, y_train, x_test, y_test = load_svmlight_files([open(name_train, 'rb'), open(name_test, 'rb')])
    
    load_estimator = False
    if load_estimator == True: 
        joblib.load("escores/grid_"+name_dataset+"_"+classifier) # load estimator
    else:
        if not(len(classifier.split(",")) > 1):
            escores = cv.load_escores(name_dataset, classifier, 1) # test score 0
            best_param_folds = cv.best_param_folds_no_frequency(escores, 0, metric) # best score per fold      
            estimator.set_params(**best_param_folds)
        estimator.fit(x_train, y_train)
                
    y_pred = estimator.predict(x_test)     
    cv.save_dict_list([y_test], [y_pred], 'y_pred/'+name_dataset+"_" +classifier +"_" +metric +"_" +cv.name_file(name_test))
    

ini = timeit.default_timer() # Time process
SEED=42
numpy.random.seed(seed=SEED)
name_dataset = sys.argv[1]
name_train=sys.argv[2]
name_test=sys.argv[3]
classifier=sys.argv[4]
predict_classifier(name_dataset, name_train, classifier, name_test, sys.argv[5])
print("Time End: %f" % (timeit.default_timer() - ini))
