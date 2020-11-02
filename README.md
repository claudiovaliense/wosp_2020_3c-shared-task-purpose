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

# Citation (Bibtex)
<pre>
@inproceedings{de-andrade-goncalves-2020-combining,
    title = "Combining Representations For Effective Citation Classification",
    author = "de Andrade, Claudio Mois{\'e}s Valiense  and
      Gon{\c{c}}alves, Marcos Andr{\'e}",
    booktitle = "Proceedings of the 8th International Workshop on Mining Scientific Publications",
    month = "05 " # aug,
    year = "2020",
    address = "Wuhan, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.wosp-1.8",
    pages = "54--58",
    abstract = "In this paper, we describe our participation in two tasks organized by WOSP 2020, consisting of classifying the context of a citation (e.g., background, motivational, extension) and whether a citation is influential in the work (or not). Classifying the context of an article citation or its influence/importance in an automated way presents a challenge for machine learning algorithms due to the shortage of information and inherently ambiguity of the task. Its solution, on the other hand, may allow enhanced bibliometric studies. Several text representations have already been proposed in the literature, but their combination has been underexploited in the two tasks described above. Our solution relies exactly on combining different, potentially complementary, text representations in order to enhance the final obtained results. We evaluate the combination of various strategies for text representation, achieving the best results with a combination of TF-IDF (capturing statistical information), LDA (capturing topical information) and Glove word embeddings (capturing contextual information) for the task of classifying the context of the citation. Our solution ranked first in the task of classifying the citation context and third in classifying its influence.",
}
</pre>
