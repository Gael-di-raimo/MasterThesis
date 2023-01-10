import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random


from utility.utility_plot import  plot_per_b_size, plotRegressions, plotIntersections, plot_per_b_size_folds
from utility.load_dt import load_dt_model
from utility.my_k_fold_cv import load_folds, my_cv_score

from modelClass.LR3M import LR3M

from sklearn.metrics import r2_score, mean_absolute_percentage_error, balanced_accuracy_score
from sklearn.base import clone


# --------------------------------------------------------------------
#  Create and test the models LRNNZX.
#	LRNNZX (a) is a model that takes as input the number
#	of non-zero elements of the solution and predicts
#	the time taken for an algorithm. LRNNZX (b) and (c) are another 
#	model that will take also the size of b. LRNNZX (c) taking x.nnz^2
#	to predict the time of the 2phase and  x.nnz*lo10(x.nnz) for the 
#	1phase.
#
#	This file wil read the folds "datasets/mydt/folds" and 
# 	plot a series of figures comparing LRNNZx (a), (b) and (c) in
#	plots/compareLRNNZX
# --------------------------------------------------------------------

def fct_general(x):
	return x

def fct_2phase(x):
	return np.square(x)

def fct_1phase(x):
	return np.multiply(x,np.log10(x))

if __name__ == "__main__":

	datasetPath = "datasets/mydt/folds"
	
	figPath = "plots/"

	if not os.path.isdir(figPath+"/compareLRNNZX"):
		os.mkdir(figPath+"/compareLRNNZX")

	figPath = figPath+"/compareLRNNZX"

	model_names = []

	#training LRNNZX (a)

	print("Starting the training and testing of LRNNZX (a)")
	featureIndicesL = np.array([4,1])
	featureIndicesU = np.array([8,5])
	sizebIndicesL = np.array([1])
	sizebIndicesU = np.array([5])

	yIndicesL = np.array([11, 12, 13])
	yIndicesU = np.array([14, 15, 16])
	
	featureNames = ["Number of non-zero elements in x",
					"Number of non-zero elements in x",
					"Number of non-zero elements in x"]

	list_folds, time_per_fold = load_folds(datasetPath, featureIndicesL, featureIndicesU, yIndicesL, yIndicesU, return_times = True)
	list_folds_size_b = load_folds(datasetPath, sizebIndicesL, sizebIndicesU, yIndicesL, yIndicesU)

	#use 7 folds to train the model and 3 folds to test for some plots

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


	#training LRNNZX (a)
	model_names.append("LNNZX (a)")

	LRNNZXa = LR3M("LNNZX (a)", featureNames)

	fold_pred_algo_a, fold_best_algo = LRNNZXa.pred_time_cv(list_folds, [0, 0, 0])

	

	LRNNZXa.regGeneral.fit(x_train, y_train[:,0])
	LRNNZXa.reg2Phase.fit(x_train, y_train[:,1])
	LRNNZXa.reg1Phase.fit(x_train, y_train[:,2])
	
	print("Making the intersection plot")
	plotIntersections(LRNNZXa, x_train, x_train, x_train, figPath)
	print("Making the regressions plot")
	plotRegressions(LRNNZXa, x_train, x_train, x_train, y_train[:,0], y_train[:,1], y_train[:,2], figPath)
	

	#training LRNNZX (b)

	print("Starting the training and testing of LRNNZX (b)")

	featureNames = ["Size of b",
					"Number of non-zero elements in x",
					"Number of non-zero elements in x"]
	
	model_names.append("LNNZX (b)")

	LRNNZXb = LR3M("LNNZX (b)", featureNames)

	fold_pred_algo_b, fold_best_algo = LRNNZXb.pred_time_cv(list_folds, [1, 0, 0])

	LRNNZXb.regGeneral.fit(x_train_size_b, y_train[:,0])
	LRNNZXb.reg2Phase.fit(x_train, y_train[:,1])
	LRNNZXb.reg1Phase.fit(x_train, y_train[:,2])
	
	print("Making the intersection plot")
	plotIntersections(LRNNZXb, None, x_train, x_train, figPath)
	print("Making the regressions plot")
	plotRegressions(LRNNZXb, x_train_size_b, x_train, x_train, y_train[:,0], y_train[:,1], y_train[:,2], figPath)
	

	#training LRNNZX (c)

	print("Starting the training and testing of LRNNZX (c)")

	featureNames = ["Size of b",
					"Number of non-zero elements in x",
					"Number of non-zero elements in x"]

	model_names.append("LNNZX (c)")

	LRNNZXc = LR3M("LNNZX (c)", featureNames)

	fold_pred_algo_c, fold_best_algo = LRNNZXc.pred_time_cv(list_folds, [1, 0, 0], fct_to_apply = [fct_general, fct_2phase, fct_1phase])

	LRNNZXc.regGeneral.fit(x_train_size_b, y_train[:,0])
	LRNNZXc.reg2Phase.fit(x_train, y_train[:,1])
	LRNNZXc.reg1Phase.fit(x_train, y_train[:,2])
	

	print("Making the regressions plot")
	plotRegressions(LRNNZXc, x_train_size_b, x_train, x_train, y_train[:,0], y_train[:,1], y_train[:,2], figPath, fct_to_apply = [fct_general, fct_2phase, fct_1phase])

	print("Making the intersection plot")
	plotIntersections(LRNNZXc, None, x_train, x_train, figPath, fct_to_apply = [fct_general, fct_2phase, fct_1phase])
	
	#plt the bar plot per b size
	dtTest = np.loadtxt("datasets/mydt/dt_test.csv", delimiter = ",")
	
	#compare LRNNZX (a), (b) and (c)

	predList = []
	predList.append(fold_pred_algo_a)
	predList.append(fold_pred_algo_b)
	predList.append(fold_pred_algo_c)

	plot_per_b_size_folds(predList, list_folds, list_folds_size_b, time_per_fold, model_names, figPath+"/LRNNZX_")

	#remake LRNNZB (b) to compare to LRNNZX (c)

	featureIndicesL = np.array([3,1])
	featureIndicesU = np.array([7,5])

	list_folds2, time_per_fold = load_folds(datasetPath, featureIndicesL, featureIndicesU, yIndicesL, yIndicesU, return_times = True)
	LRNNZBb = LR3M("LNNZB (b)", featureNames)
	indices_used_per_method = [1,0,0]
	
	fold_pred_algo_d, fold_best_algo = LRNNZBb.pred_time_cv(list_folds2, indices_used_per_method, modelName = "LRNNZBb")
	
	predList = []
	predList.append(fold_pred_algo_c)
	predList.append(fold_pred_algo_d)
	predList.append(fold_best_algo)


	model_names = []
	model_names.append("LNNZX (c)")
	model_names.append("LNNZB (b)")
	model_names.append("GT")

	plot_per_b_size_folds(predList, list_folds, list_folds_size_b, time_per_fold, model_names, figPath+"/LRNNZXvsLRNNZB_")

	testValues = True
	

	if testValues:
		
		x_list = []
		
		for fold in list_folds:
			x_list.append(fold[0])

		x = np.concatenate(x_list)

		mask = x[:, 0] < 200

		print("%f percent of the samples in the samples dedicated to the cross validation has a x.nnz < 200 "%(len(x[mask,:])/len(x)*100))

		print("Coef of LRNNZX (a) 2phase reg is %f with offset of %f" % (LRNNZXa.reg2Phase.coef_, LRNNZXa.reg2Phase.intercept_))
		print("Coef of LRNNZX (a) 1phase reg is %f with offset of %f" % (LRNNZXa.reg1Phase.coef_, LRNNZXa.reg1Phase.intercept_))
		x_intersec = (LRNNZXa.reg2Phase.intercept_ - LRNNZXa.reg1Phase.intercept_)/(LRNNZXa.reg1Phase.coef_-LRNNZXa.reg2Phase.coef_)

		mask = x[:, 0] < x_intersec
		print("Intersection is in x.nnz = %f and %f percent of the sample are below"%(x_intersec,len(x[mask,:])/len(x)*100))

