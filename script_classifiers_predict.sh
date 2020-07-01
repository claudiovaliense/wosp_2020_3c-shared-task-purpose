predict(){
	for index in $(seq 0 $4); do									
			python3.6 classifiers_predict.py $1 dataset/representations/$1/train$index dataset/representations/$1/test$index $2 $3 > y_pred/$1_nohup_$2_$3_predict_test$index.txt &
	done
}

predict 3c-shared-task-purpose passive_aggressive f1_macro 0
#predict 3c-shared-task-purpose sgd f1_macro 0
#predict 3c-shared-task-purpose svm f1_macro 0