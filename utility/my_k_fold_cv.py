import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import KFold



def my_cv_score(list_folds, model, list_score_fct, return_t_pred = False):
	# --------------------------------------------------------------------
	# Fit a model with the list of folds with cross validation anc compute
	# 	a list of returned score on train and test folds:
	#	:param list_folds: list np array each being a fold, a fold 
	#					   contains feature and prediction in np array
	#	:param model: the model
	#	:param list_score_fct: list of function that will be used to
	#							evaluate model. Input should be in the
	#							form of (y_true, ypred)
	#	:return: score_train, score_test, return_t_pred
	# --------------------------------------------------------------------

	scores_test = np.zeros((len(list_folds),len(list_score_fct)))
	scores_train = np.zeros((len(list_folds),len(list_score_fct)))

	t_pred_list = []

	for fold_nb in range(len(list_folds)):
		#clone model to avoid using/fit model on multiple fold iteration
		tmp_model = clone(model)
		
		fold_test = list_folds[fold_nb]

		x_list = []
		y_list = []

		for fold_nb_2 in range(len(list_folds)):

			if fold_nb_2 != fold_nb:
				fold = list_folds[fold_nb_2]
				x_list.append(fold[0])
				y_list.append(fold[1])

		x_train = np.concatenate(x_list)

		y_train = np.concatenate(y_list)

		x_test = fold_test[0]
		y_test = fold_test[1]
		
		y_train = np.ravel(y_train)
		y_test = np.ravel(y_test)

		tmp_model.fit(x_train, y_train)

		t0 = time.time_ns()
		y_pred_test = tmp_model.predict(x_test)
		t1 = time.time_ns()

		t_pred = (t1 - t0)

		y_pred_train = tmp_model.predict(x_train)

		for score_fct_nb in range(len(list_score_fct)):

			score_fct = list_score_fct[score_fct_nb]

			scores_test[fold_nb, score_fct_nb] = score_fct(y_test, y_pred_test)
			scores_train[fold_nb, score_fct_nb] = score_fct(y_train, y_pred_train)

		#reinsert test fold
		t_pred_list.append(t_pred)
	
	if return_t_pred:
		return np.mean(scores_train, axis = 0), np.mean(scores_test, axis = 0), t_pred_list
	else:
		return np.mean(scores_train, axis = 0), np.mean(scores_test, axis = 0)

def load_folds(dt_path, x_indices_L, x_indices_U, y_indices_L, y_indices_U, return_times = False):

	list_folds = []
	list_folds_times = []
	
	for filename in os.listdir(dt_path):

		if filename.find("fold") != -1:
			print("Loading  folds %s" % (dt_path+"/"+filename))
			dt_fold = np.loadtxt(dt_path+"/"+filename, delimiter=",")

			x = np.concatenate((dt_fold[:, x_indices_L], dt_fold[:, x_indices_U]))
			y = np.concatenate((dt_fold[:, y_indices_L], dt_fold[:, y_indices_U]))

			times = np.concatenate((dt_fold[:, np.array([11, 12, 13, 28, 29, 30, 31])], dt_fold[:, np.array([14, 15, 16, 32, 33, 34, 35])]))

			list_folds.append([x, y])
			list_folds_times.append(times)

	if len(list_folds) == 0:
		print("No fold found on path %s" %(dt_path))
	
	if return_times:
		return list_folds, list_folds_times
	else:
		return list_folds 

def load_test(dt_path, x_indices_L, x_indices_U, y_indices_L, y_indices_U, return_times = False):

	test_set = []
	test_set_times = []
	
	for filename in os.listdir(dt_path):

		if filename.find("test") != -1:
			print("Loading  test set %s" % (dt_path+"/"+filename))
			dt_test = np.loadtxt(dt_path+"/"+filename, delimiter=",")

			x = np.concatenate((dt_test[:, x_indices_L], dt_test[:, x_indices_U]))
			y = np.concatenate((dt_test[:, y_indices_L], dt_test[:, y_indices_U]))

			times = np.concatenate((dt_test[:, np.array([11, 12, 13, 28, 29, 30, 31])], dt_test[:, np.array([14, 15, 16, 32, 33, 34, 35])]))

			test_set = [x, y]
			test_set_times = times

			break

	if len(test_set) == 0:
		print("No test set found on path %s" %(dt_path))
	
	if return_times:
		return test_set, test_set_times
	else:
		return test_set 


if __name__ == "__main__":

	make_k_fold("../datasets/mydt/", 10, 0.8, 0.05)


