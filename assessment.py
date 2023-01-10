import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random


from utility.utility_plot import  plot_per_b_size, plotRegressions, plotIntersections, plot_per_b_size_folds
from utility.load_dt import load_dt_model
from utility.my_k_fold_cv import load_folds, my_cv_score, load_test
from utility.metrics import time_metrics_folds, getFeaturesComputationTimeFold
from utility.utility_tex_table import export_tex_table

from modelClass.Reg3MWithProxy import R3MwProxy

from sklearn.metrics import r2_score, mean_absolute_percentage_error, balanced_accuracy_score
from sklearn.base import clone

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from modelClass.RFAdaProxy import RFAdaProxy
from modelClass.LR3M import LR3M

# --------------------------------------------------------------------
# Train the model RFAdaProxy (b) with all the 10 fols to finaly
#	test its performances against LRNNZB (b) on the test set. Then,
#	train with all the dataset and save the model in
#	'model/RFAdaProxy_final.sav'
# --------------------------------------------------------------------

if __name__ == "__main__":

	datasetPath = "datasets/mydt/folds"


	

	featureIndicesL = np.array([0,1,2,3,20,21,22,23],)
	featureIndicesU = np.array([0,5,6,7,24,25,26,27])

	sizebIndicesL = np.array([1])
	sizebIndicesU = np.array([5])


	yIndicesL = np.array([11, 12, 13, 4])
	yIndicesU = np.array([14, 15, 16, 8])

	#loading train and test set
	
	test_set, test_set_times = load_test(datasetPath, featureIndicesL, featureIndicesU, yIndicesL, yIndicesU, return_times = True)
	


	index_reg_general = np.array([1])
	index_reg_2phase = np.arange(9)
	index_reg_1phase = np.arange(9)

	index_reg_proxy = np.arange(8)

	RFAda = RFAdaProxy()

	LRNNZBb = LR3M("LRNNZBb", None)

	if not os.path.isdir("model"):
		os.mkdir("model")


	if os.path.isfile("model/RFAdaProxy.sav"):
		RFAda.loadModel("model")

	if os.path.isfile("model/LRNNZBb.sav"):
		LRNNZBb.loadModel("model")
	

	#train models if not already trained
	if not os.path.isfile("model/RFAdaProxy.sav") or not os.path.isfile("model/LRNNZBb.sav"):
		
		folds, time_per_fold = load_folds(datasetPath, featureIndicesL, featureIndicesU, yIndicesL, yIndicesU, return_times = True)

		x_list = []
		y_list = []
		time_list = []
		for j in range(len(folds)):
			fold = folds[j]
			x = fold[0]

			x_list.append(x)
			y_list.append(fold[1])


		#merge training folds
		x_train = np.concatenate(x_list).astype(int)
		y_train = np.concatenate(y_list).astype(int)

		if not os.path.isfile("model/RFAdaProxy.sav"):
			RFAda.fit(x_train, y_train)
			RFAda.saveModel("model")

		if not os.path.isfile("model/LRNNZBb.sav"):
			print("Fitting LRNNZBb")
			LRNNZBb.fit(x_train, y_train)
			print("saving LRNNZBb")
			LRNNZBb.saveModel("model")

	#test the model
	x_test = test_set[0].astype(int)
	y_test = test_set[1].astype(int)

	t0 = time.time_ns()
	pred_algo = RFAda.predict(x_test)
	t1 = time.time_ns()

	t_pred = t1 - t0

	t_feature = np.sum(getFeaturesComputationTimeFold(test_set_times)/(10**7))


	algo_times = y_test[:, np.arange(3)]
	best_algo = np.argmin(algo_times, axis = 1)

	pred_algo_LR = LRNNZBb.predict(x_test)

	time_best_algo = np.sum(algo_times[np.arange(0, algo_times.shape[0]), best_algo])/(10**7)
	time_pred_algo = np.sum(algo_times[np.arange(0, algo_times.shape[0]), pred_algo])/(10**7)

	time_pred_algo_LR =  np.sum(algo_times[np.arange(0, algo_times.shape[0]), pred_algo_LR])/(10**7)

	#add feature time + time prediction 
	total_computation_time_pred_algo = t_pred/10**9 + time_pred_algo + t_feature


	balanced_accuracy = balanced_accuracy_score(best_algo, pred_algo)
	balanced_accuracy_2 = balanced_accuracy_score(best_algo, pred_algo_LR)

	print("The total computation time by predicted algo (RFAdaProxy) over the test set is %fs" %(total_computation_time_pred_algo))
	print("The time taken for the prediction was %fs" %( t_pred/10**9 + t_feature))

	print("The total computation time by predicted algo (LRNNZb) over the test set is %fs" %(time_pred_algo_LR))
	print("The time of the best algo would be %fs" % (time_best_algo))
	print("The balanced_accuracy over the test set is %f " %(balanced_accuracy))
	print("The balanced_accuracy over the test set is %f " %(balanced_accuracy_2))


	#train Final model on the full dataset


	

	if not os.path.isfile("model/RFAdaProxy_final.sav"):

		RFAda = RFAdaProxy()
		folds, time_per_fold = load_folds(datasetPath, featureIndicesL, featureIndicesU, yIndicesL, yIndicesU, return_times = True)

		x_list = []
		y_list = []
		time_list = []
		for j in range(len(folds)):
			fold = folds[j]
			x = fold[0]

			x_list.append(x)
			y_list.append(fold[1])


		#merge training folds
		x_train = np.concatenate(x_list).astype(int)
		y_train = np.concatenate(y_list).astype(int)

		x_train = np.concatenate((x_train, x_test))
		y_train = np.concatenate((y_train, y_test))
		

		RFAda.fit(x_train, y_train)
		RFAda.saveModel("model")
