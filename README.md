# wosp_2020_3c-shared-task-purpose

# Author: Claudio Moisés Valiense de Andrade

# Download Glove 6B tokens. Place the extracted file in a directory above the project directory
https://nlp.stanford.edu/projects/glove/

# Install python 3.6
sudo apt-get install python3.6

# Prerequisite module
python3.6 -m pip install sklearn pandas heapq_max nltk matplotlib numpy unidecode pyjarowinkler vaderSentiment gensim imblearn

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

# We create an image through the docker with the entire process configured
https://hub.docker.com/r/claudiovaliense/wosp\_2020\_3c-shared-task-purpose
