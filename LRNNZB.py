import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random


from utility.utility_plot import  plot_per_b_size, plotRegressions, plotIntersections, plot_per_b_size_folds
from utility.load_dt import load_dt_model
from utility.my_k_fold_cv import load_folds, my_cv_score

from sklearn.metrics import r2_score, mean_absolute_percentage_error, balanced_accuracy_score
from sklearn.base import clone


from modelClass.LR3M import LR3M

# -----------------------------------------------------------
#  Create and test the models LRNNZB.
#	LRNNZB (a) is a model that takes as input the number
#	of non-zero elements of the right-hand side and predicts
#	the time taken for an algorithm. LRNNZB (b) is another 
#	model that will take also the size of b.
#
#	This file wil read the folds "datasets/mydt/folds" and 
# 	plot a series of figures comparing LRNNZB (a) and (b) in
#	plots/compareLRNNZB
# -----------------------------------------------------------


def pred_time_cv(folds, indices_used_per_method, modelGeneral, model2Phase, model1Phase, modelName = None):
	
	r2_train_folds = np.zeros((len(folds), 3))
	r2_test_folds = np.zeros((len(folds), 3))

	bas = np.zeros(len(folds))

	fold_model_choice_time = np.zeros(len(folds))
	fold_best_choice_time = np.zeros(len(folds))
	fold_average_choice_time = np.zeros(len(folds))
	fold_choose_1phase_time = np.zeros(len(folds))
	fold_pred_time = np.zeros(len(folds))


	fold_pred_algo = []
	fold_best_algo = []



	for i in range(len(folds)):

		print("Fold %d"%(i))
		modelCloneGeneral = clone(modelGeneral)
		modelClone2Phase = clone(model2Phase)
		modelClone1Phase = clone(model1Phase)


		fold_test = folds[i]

		x_list_general = []
		x_list_2phase = []
		x_list_1phase = []
		y_list = []

		for j in range(len(folds)):
			if j != i:
				fold = folds[j]
				x = fold[0]

				x_list_general.append(x[:, indices_used_per_method[0]])
				x_list_2phase.append(x[:, indices_used_per_method[1]])
				x_list_1phase.append(x[:, indices_used_per_method[2]])

				y_list.append(fold[1])

		x_train_general = np.concatenate(x_list_general)
		
		
		x_train_2phase = np.concatenate(x_list_2phase)

		

		x_train_1phase = np.concatenate(x_list_1phase)
		
		if len(x_train_general.shape) == 1:
			x_train_general = np.expand_dims(x_train_general, 1)

		if len(x_train_2phase.shape) == 1:
			x_train_2phase = np.expand_dims(x_train_2phase, 1)

		if len(x_train_1phase.shape) == 1:
			x_train_1phase = np.expand_dims(x_train_1phase, 1)

		y_train = np.concatenate(y_list)
		
		y_test = fold_test[1]
		x_test = fold_test[0]

		x_test_general  = x_test[:, indices_used_per_method[0]]
		x_test_2phase  = x_test[:, indices_used_per_method[1]]
		x_test_1phase  = x_test[:, indices_used_per_method[2]]

		if len(x_test_general.shape) == 1:
			x_test_general = np.expand_dims(x_test_general, 1)

		if len(x_test_2phase.shape) == 1:
			x_test_2phase = np.expand_dims(x_test_2phase, 1)

		if len(x_test_1phase.shape) == 1:
			x_test_1phase = np.expand_dims(x_test_1phase, 1)


		y_train_pred = np.zeros((y_train.shape[0], y_train.shape[1]))
		y_test_pred = np.zeros((y_test.shape[0], y_test.shape[1]))


		modelCloneGeneral.fit(x_train_general, y_train[:,0])
		modelClone2Phase.fit(x_train_2phase, y_train[:,1])
		modelClone1Phase.fit(x_train_1phase, y_train[:,2])


		y_train_pred[:,0] = modelCloneGeneral.predict(x_train_general)
		y_train_pred[:,1] = modelClone2Phase.predict(x_train_2phase)
		y_train_pred[:,2] = modelClone1Phase.predict(x_train_1phase)

		t0 = time.time_ns()
		y_test_pred[:,0] = modelCloneGeneral.predict(x_test_general)
		y_test_pred[:,1] = modelClone2Phase.predict(x_test_2phase)
		y_test_pred[:,2] = modelClone1Phase.predict(x_test_1phase)
		t1 = time.time_ns()

		t_pred = (t1-t0)/(10**9)

		fold_pred_time[i] = t_pred

		r2_train_folds[i, 0] = r2_score(y_train[:,0], y_train_pred[:,0])
		r2_train_folds[i, 1] = r2_score(y_train[:,1], y_train_pred[:,1])
		r2_train_folds[i, 2] = r2_score(y_train[:,2], y_train_pred[:,2])

		r2_test_folds[i, 0] = r2_score(y_test[:,0], y_test_pred[:,0])
		r2_test_folds[i, 1] = r2_score(y_test[:,1], y_test_pred[:,1])
		r2_test_folds[i, 2] = r2_score(y_test[:,2], y_test_pred[:,2])


		pred_algo =  np.argmin(y_test_pred, axis=1)


		random_algo = np.mean(y_test, axis=1)
		best_algo = np.argmin(y_test, axis=1)
		choose_1phase = np.ones(best_algo.shape)
		choose_1phase = np.multiply(choose_1phase,2).astype(int)
		

		fold_model_choice_time[i] = np.sum(y_test[np.arange(0, y_test.shape[0]), pred_algo])
		fold_average_choice_time[i] = np.sum(random_algo)
		fold_best_choice_time[i] =  np.sum(y_test[np.arange(0, y_test.shape[0]), best_algo])
		fold_choose_1phase_time[i] =  np.sum(y_test[np.arange(0, y_test.shape[0]), choose_1phase])

		fold_pred_algo.append(pred_algo)
		fold_best_algo.append(best_algo)
		bas[i] = balanced_accuracy_score(best_algo, pred_algo)


	r2_train = np.mean(r2_train_folds, axis = 0)
	r2_test = np.mean(r2_test_folds, axis = 0)
	
	model_choice_time = np.mean(fold_model_choice_time)
	average_choice_time = np.mean(fold_average_choice_time)
	best_choice_time = np.mean(fold_best_choice_time)
	onephase_choice_time = np.mean(fold_choose_1phase_time)
	average_pred_time = np.mean(fold_pred_time)


	if not(modelName is None):
		print("---------------%s---------------" % (modelName))
	else:
		print("-----------------------------------")

	print("With regression model %s"%(type(modelGeneral).__name__))
	print("R2 score train = %s" % (r2_train))
	print("R2 score test = %s"% (r2_test))

	print("Model choice time (test) = %f s" % (model_choice_time/(10**7)))
	print("Model choice time (test) + pred time = %f s" % (model_choice_time/(10**7) + average_pred_time))
	print("Best choice time = %f s"% (best_choice_time/(10**7)))
	print("Average choice time = %f s"% (average_choice_time/(10**7)))
	print("One phase choice time = %f s"% (onephase_choice_time/(10**7)))
	print("average balanced accuracy  = %f"% (np.mean(bas)))

	if not(modelName is None):
		print("---------------%s---------------" % (modelName))
	else:
		print("-----------------------------------")



	return fold_pred_algo, fold_best_algo

if __name__ == "__main__":

	
	datasetPath = "datasets/mydt/folds"

	figPath = "plots"

	if not os.path.isdir(figPath+"/compareLRNNZB"):
		os.mkdir(figPath+"/compareLRNNZB")

	figPath = figPath+"/compareLRNNZB"

	modelNames = []

	#training LRNNZB (a)

	print("Starting the training and testing of LRNNZB (a)")
	featureIndicesL = np.array([3,1])
	featureIndicesU = np.array([7,5])
	sizebIndicesL = np.array([1])
	sizebIndicesU = np.array([5])

	yIndicesL = np.array([11, 12, 13])
	yIndicesU = np.array([14, 15, 16])

	featureNames = ["Number of non-zero elements in b",
					"Number of non-zero elements in b",
					"Number of non-zero elements in b"]

	#loads the folds
	list_folds, time_per_fold = load_folds(datasetPath, featureIndicesL, featureIndicesU, yIndicesL, yIndicesU, return_times = True)
	list_folds_size_b = load_folds(datasetPath, sizebIndicesL, sizebIndicesU, yIndicesL, yIndicesU)


	#only take the time of the algorithm and not the feature computation time
	for i in range(len(time_per_fold)):
		
		fold_time = time_per_fold[i]

		time_per_fold[i] = fold_time[:, np.array([0,1,2])]
	

	indices_used_per_method = [0, 0, 0]

	#prepare the dataset for intersection plots, etc
	#that take 7 folds for training and 3 folds for testing

	list_x_test = []
	list_y_test = []
	list_x_train = []
	list_y_train = []

	for i in range(len(list_folds)):

		fold = list_folds[i]
		if i >=7:
			list_x_test.append(fold[0])
			list_y_test.append(fold[1])
		else:
			list_x_train.append(fold[0])
			list_y_train.append(fold[1])

	x_test = np.concatenate(list_x_test)
	x_train = np.concatenate(list_x_train)

	x_test_size_b = x_test[:,1]
	x_test = x_test[:,0]
	
	x_train_size_b = x_train[:,1]
	x_train = x_train[:,0]


	x_test = np.expand_dims(x_test, 1)
	x_test_size_b = np.expand_dims(x_test_size_b, 1)
	
	x_train = np.expand_dims(x_train, 1)
	x_train_size_b = np.expand_dims(x_train_size_b, 1)

	y_test = np.concatenate(list_y_test)
	y_train = np.concatenate(list_y_train)

	#train LRNNZB (a)
	modelNames.append("LNNZB (a)")

	LRNNZBa = LR3M("LNNZB (a)", featureNames)

	fold_pred_algo_a, fold_best_algo = LRNNZBa.pred_time_cv(list_folds, indices_used_per_method, modelName = "LRNNZBa")


	LRNNZBa.regGeneral.fit(x_train, y_train[:,0])
	LRNNZBa.reg2Phase.fit(x_train, y_train[:,1])
	LRNNZBa.reg1Phase.fit(x_train, y_train[:,2])
	
	print("Making the intersection plot")
	plotIntersections(LRNNZBa, x_train, x_train, x_train, figPath)

	print("Making the regressions plot")
	plotRegressions(LRNNZBa, x_train, x_train, x_train, y_train[:,0], y_train[:,1], y_train[:,2], figPath)
	

	#train LRNNZB (b)
	LRNNZBb = LR3M("LNNZB (b)", featureNames)
	indices_used_per_method = [1,0,0]
	fold_pred_algo_b, fold_best_algo = LRNNZBb.pred_time_cv(list_folds, indices_used_per_method, modelName = "LRNNZBb")

	LRNNZBb.regGeneral.fit(x_train_size_b, y_train[:,0])
	LRNNZBb.reg2Phase.fit(x_train, y_train[:,1])
	LRNNZBb.reg1Phase.fit(x_train, y_train[:,2])
	
	print("Making the intersection plot")
	plotIntersections(LRNNZBb, None, x_train, x_train, figPath)

	print("Making the regressions plot")
	plotRegressions(LRNNZBb, x_train_size_b, x_train, x_train, y_train[:,0], y_train[:,1], y_train[:,2], figPath)

	predList = []
	predList.append(fold_pred_algo_a)
	predList.append(fold_pred_algo_b)
	predList.append(fold_best_algo)

	

	model_names = []
	model_names.append('LRNNZBa')
	model_names.append('LRNNZBb')
	model_names.append('GT')
	plot_per_b_size_folds(predList, list_folds, list_folds_size_b, time_per_fold, model_names, figPath+"/LRNNZB_")