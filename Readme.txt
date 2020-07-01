# Author: Claudio Moisés Valiense de Andrade

# Install python 3.6
sudo apt-get install python3.6

# Prerequisite module
python3.6 -m pip install sklearn pandas heapq_max nltk matplotlib numpy unidecode pyjarowinkler vaderSentiment gensim                                      

# Create file in the format that contains the titles and context of the description
python3.6 create_text.py

# Create representation
sh script_representation_tfidf.sh 

# Seeks the best parameter for the classifier (Optimize hyperparameter)
sh script_classifiers_score.sh 

# Predict the classes using the parameter best evaluated in the previous step
sh script_classifiers_predict.sh

# Create file for submission
python3.6 create_file_submission.py
