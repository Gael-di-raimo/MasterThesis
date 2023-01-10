import numpy as np
import os
import time

import matplotlib.pyplot as plt
import random


from utility.utility_plot import  plot_per_b_size, plotRegressions, plotIntersections, plot_MAPE_xnnz_per_b_size, plot_MAPE_xnnz_per_xnnz
from utility.load_dt import load_dt_NNZXreg, load_dt_NNZXreg2
from utility.metrics import median_absolute_percentage_error, time_metrics, time_metrics_folds, getFeaturesComputationTimeFold
from utility.utility_tex_table import export_tex_table


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.base import clone

from sklearn.ensemble import IsolationForest

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from yellowbrick.regressor import ResidualsPlot, residuals_plot

from utility.my_k_fold_cv import load_folds, my_cv_score

# --------------------------------------------------------------------
# This file will try different model for the regression of 
# 	the number of non-zero elements of x. This model will further be
#	used to predict the algorithm computation times.
# --------------------------------------------------------------------



def getRegModel(modelName):

	if modelName == "LinearRegression":
		return LinearRegression()#n_jobs = -1)

	if modelName == "DecisionTree":
		return DecisionTreeRegressor(random_state = 165391, max_depth = 20)

	if modelName == "MLP":
		#reg = MLPRegressor(random_state = 165391, learning_rate_init=0.01, max_iter = 200, batch_size = 200)
		reg = MLPRegressor(random_state = 165391, hidden_layer_sizes = 10, max_iter = 600)
		return make_pipeline(StandardScaler(), reg)

	if modelName == "KNN":
		return KNeighborsRegressor()

	if modelName == "RandomForest":
		#return RandomForestRegressor(random_state = 165391, n_estimators = 5)#, n_jobs = -1)

		return RandomForestRegressor(random_state = 165391)#, n_estimators = 2, max_depth = 20)#, n_jobs = -1)

	if modelName == "ExtraTrees":
		return ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 50)#, n_jobs = -1)

	if modelName == "ADABoost":
		return AdaBoostRegressor(random_state = 165391)

	if modelName == "Bagging":
		return BaggingRegressor()#n_jobs = -1)

	if modelName == "DummyMean":
		return DummyRegressor(strategy= 'mean')
	
	if modelName == "DummyMedian":
		return DummyRegressor(strategy= 'median')

	if modelName == "myDummy":
		return myDummyRegressor()



def getTotalComputationTime(dt):

	timeL = np.arange(11, 14)
	timeU = np.arange(14, 17)

	yTest = np.concatenate((dt[:, timeL], dt[:, timeU]))

	bestAlgo = np.argmin(yTest, axis=1)

	
	return np.sum(yTest[np.arange(0, yTest.shape[0]), bestAlgo])/(10**7)

def getBestMethodComputationTime(dt):	
	# --------------------------------------------------------------------
	#  Return the time each sample of the dt will take
	# to solve the system if the best algorithm is chosen
	# --------------------------------------------------------------------


	timeL = np.arange(11, 14)
	timeU = np.arange(14, 17)

	y = np.concatenate((dt[:, timeL], dt[:, timeU]))

	bestAlgo = np.argmin(y, axis=1)

	
	return y[np.arange(0, y.shape[0]), bestAlgo]

def getMethodComputationTimePerOrder(dt):
	# --------------------------------------------------------------------
	#  Return the time in 100 ns each sample of the dt will take
	# to solve the system if it take the best algorithm the second best 
	# and the worst.
	#
	#	return: timeBestAlgo, timeSecondBestAlgo, timeWorstAlgo
	# --------------------------------------------------------------------

	timeL = np.arange(11, 14)
	timeU = np.arange(14, 17)

	y = np.concatenate((dt[:, timeL], dt[:, timeU]))

	worstAlgo = np.argmax(y, axis=1)
	bestAlgo = np.argmin(y, axis=1)

	tmp_y = y.copy()
	tmp_y[np.arange(0, y.shape[0]),worstAlgo] = -1
	tmp_y[np.arange(0, y.shape[0]),bestAlgo] = -1

	secondBestAlgo = np.argmax(tmp_y, axis = 1)


	timeBestAlgo = y[np.arange(0, y.shape[0]), bestAlgo]
	timeSecondBestAlgo = y[np.arange(0, y.shape[0]), secondBestAlgo]
	timeWorstAlgo = y[np.arange(0, y.shape[0]), worstAlgo]

	return timeBestAlgo, timeSecondBestAlgo, timeWorstAlgo

def getFeaturesComputationTime(dt, featureIndicesL =  np.array([20,21,22,23])):

	mask = featureIndicesL >= 20

	featureWithTime = featureIndicesL[mask]
	
	if len(featureWithTime) == 0:
		return 0

	featureWithTime  = featureWithTime - 20

	indicesTimeL = featureWithTime + 28
	indicesTimeU = featureWithTime + 32

	timeL = np.sum(dt[:, indicesTimeL])/(10**7)
	timeU = np.sum(dt[:, indicesTimeU])/(10**7)

	return timeL + timeU


def split_train_test_fold(folds, nb_fold_in_train):

	x_list_train = []
	y_list_train = []

	x_list_test = []
	y_list_test = []

	for i in range(len(folds)):
		
		fold = folds[i]
		
		if i >= nb_fold_in_train:
			x_list_test.append(fold[0])
			y_list_test.append(fold[1])
		else:
			x_list_train.append(fold[0])
			y_list_train.append(fold[1])


	x_train = np.concatenate(x_list_train)
	y_train = np.concatenate(y_list_train)


	x_test = np.concatenate(x_list_test)
	y_test = np.concatenate(y_list_test)

	return x_train, y_train, x_test, y_test


def moreTestLR(model, folds, figPath, times):

	if not os.path.isdir(figPath+"NNZXReg/BiasVar"):
		os.mkdir(figPath+"NNZXReg/BiasVar")

	if not os.path.isdir(figPath+"NNZXReg/residual"):
		os.mkdir(figPath+"NNZXReg/residual")

	list_score_fct = [r2_score, mean_absolute_percentage_error]
	
	score_train, score_test, t_pred_folds = my_cv_score(folds, model, list_score_fct, return_t_pred = True)

	r2_score_test = score_test[0]
	r2_score_train = score_train[0]

	MAPE_test = score_test[1]
	MAPE_train = score_train[1]

	t_pred_folds = np.array(t_pred_folds, dtype=np.float64)

	t_metrics = time_metrics_folds(times, t_pred_folds)

	print("LR R^2 test = %f, R^2 train = %f , MAPE test %f, MAPE train %f, beta_12 = %f and gamma_12 = %f"%
		(r2_score_test, r2_score_train, MAPE_test, MAPE_train, t_metrics[0], t_metrics[2]))
	
	x_train, y_train, x_test, y_test = split_train_test_fold(folds, 7)

	reg = clone(model)
	reg.fit(x_train, y_train)

	visualizer =  residuals_plot(reg, x_train, y_train, x_test, y_test, is_fitted = 'True', show = False)
	visualizer.show(outpath=figPath+"NNZXReg/residual/LR_residual.png")
	
	return visualizer

def make_res_plot(figpath, model, folds, nb_fold_in_train):

	x_train, y_train, x_test, y_test = split_train_test_fold(folds, nb_fold_in_train)
	
	print(np.ravel(y_train).shape)

	reg = clone(model)
	

	y_train = np.ravel(y_train) 
	y_test = np.ravel(y_test) 

	reg.fit(x_train, y_train)

	visualizer =  residuals_plot(reg, x_train, np.ravel(y_train), x_test, np.ravel(y_test), is_fitted = 'True', show = False)

	ax = plt.gca()
	ax.set_ylim(ymin=-45000, ymax=45000)

	visualizer.show(outpath=figpath+"residual.png")


def testParamModel(model, model_name, param_name, param_values, folds, figPath, times, param_in_table = None, is_estimator_param = False):

	if param_in_table is None:
		param_in_table = param_values

	
	
	col_names = [param_name, "$R^2$ test", "$R^2$ train", "MAPE test", "MAPE train","$\\beta_{1,2}$","$\\gamma_{1,2}$"]

	
	metric_array = np.zeros((len(param_values), len(col_names)))
	metric_array[:,0] = np.array(param_values)

	#because of the pipeline
	if param_name != None and model_name == "MLP" :
		param_name = 'mlpregressor__' + param_name

	list_score_fct = [r2_score, mean_absolute_percentage_error]
	regs = []

	for i in range(len(param_values)):

		if param_name!= None:
			print("Testing %s with %s = %s" %(model_name, param_name, param_values[i]))
		else:
			print("Testing %s with default parameters")

		reg = clone(model)

		
		if	param_name!= None:

			if is_estimator_param:
				#change the parameter of the inner estimator
				param = {'base_estimator__'+param_name: param_values[i]}
			else:
				param = {param_name: param_values[i]}

			reg = reg.set_params(**param)
		
		

		#compute the different metrics


		score_train, score_test, t_pred_folds = my_cv_score(folds, reg, list_score_fct, return_t_pred = True)
		

		r2_score_test = score_test[0]
		r2_score_train = score_train[0]

		MAPE_test = score_test[1]
		MAPE_train = score_train[1]

		metric_array[i,1:3] = np.array([r2_score_test, r2_score_train])
		metric_array[i,3:5] = np.array([MAPE_test, MAPE_train])


		t_metrics = time_metrics_folds(times, t_pred_folds)
		metric_array[i,5:7] = t_metrics[np.array([0,2])]

		regs.append(reg)

	if not os.path.isdir('results'):
		os.mkdir('results')


	if model_name == "KNN":
		reg_index = max(np.argmax(metric_array[:,1]), np.argmin(metric_array[:,3]))
	else:
		reg_index_r2 = np.argmax(metric_array[:,1])
		reg_index_MAPE = np.argmin(metric_array[:,3])


	
	
	print("Visualize not working for now because there is 10 folds ...")
	
	if not os.path.isdir(figPath+"NNZXReg/BiasVar"):
		os.mkdir(figPath+"NNZXReg/BiasVar")
	

	plt.clf()
	plt.plot(param_values, metric_array[:,1], 'g', label = 'test')
	plt.plot(param_values, metric_array[:,2], 'r', label = 'train')
	plt.xlabel(param_name)
	plt.ylabel("$R^2$")
	plt.legend()
	plt.savefig(figPath+"NNZXReg/BiasVar/"+model_name+"_"+param_name+"_r2_bias_variance.png")
	

	plt.clf()
	plt.plot(param_values, metric_array[:,3], 'g', label = 'test')
	plt.plot(param_values, metric_array[:,4], 'r', label = 'train')
	plt.xlabel(param_name)
	plt.ylabel("$MAPE$")
	plt.legend()
	plt.savefig(figPath+"NNZXReg/BiasVar/"+model_name+"_"+param_name+"_MAPE_bias_variance.png")


	mask = np.zeros(len(metric_array[:,0]))

	for i in range(len(mask)):
		if metric_array[i,0] in param_in_table:
			mask[i] = 1

	mask = np.array(mask, dtype=bool)

	export_tex_table(metric_array[mask,:], col_names, 'results', model_name+"_"+param_name+'.tex', printTable = True)

	x_train, y_train, x_test, y_test = split_train_test_fold(folds, 7)
	
	print(np.ravel(y_train).shape)

	reg_r2 = clone(regs[reg_index_r2])
	reg_MAPE = clone(regs[reg_index_MAPE])

	print("%s choosen is the one with the %s = %s with highest r2." %(model_name, param_name,  param_values[reg_index_r2]))
	print("%s choosen is the one with the %s = %s with lowest MAPE." %(model_name, param_name,  param_values[reg_index_MAPE]))
	
	reg_r2.fit(x_train, y_train)

	visualizer =  residuals_plot(reg_r2, x_train, np.ravel(y_train), x_test, np.ravel(y_test), is_fitted = 'True', show = False)

	ax = plt.gca()
	ax.set_ylim(ymin=-45000, ymax=45000)

	if not os.path.isdir(figPath+"NNZXReg/residual"):
		os.mkdir(figPath+"NNZXReg/residual")

	visualizer.show(outpath=figPath+"NNZXReg/residual/"+model_name+"_"+param_name+"_r2_residual.png")

	reg_MAPE.fit(x_train, y_train)

	visualizer =  residuals_plot(reg_MAPE, x_train, np.ravel(y_train), x_test, np.ravel(y_test), is_fitted = 'True', show = False)
	visualizer.show(outpath=figPath+"NNZXReg/residual/"+model_name+"_"+param_name+"_MAPE_residual.png")
	

def testParams(modelName, model, folds, figPath, times):

	
	if modelName ==  "LinearRegression":
		return moreTestLR(model, folds, figPath, times)

	elif modelName == "DecisionTree":
		testParamModel(model, "DecisionTree", "max_depth", [10,20,30,40,50,60,70,80], folds, figPath, times)

	elif modelName == "MLP":

		param_name = 'hidden_layer_sizes'
		param_values = [10,50,100,200,500]
		
		return testParamModel(model, "MLP", param_name, param_values, folds, figPath, times)
		#print("More test not available for "+modelName)

	elif modelName == "KNN":

		testParamModel(model, "KNN", "n_neighbors", [5,8,10,50], folds, figPath, times)

	elif modelName == "RandomForest":
		#testParamModel(model, "RandomForest", "max_features", np.divide(np.arange(1,9),8), folds, figPath, times)
		#testParamModel(model, "RandomForest", "n_estimators", [10,20,50,100,200], folds, figPath, times)
		testParamModel(model, "RandomForest", "max_depth", [10, 20, 30, 50], folds, figPath, times)


	elif modelName == "ExtraTrees":
		#testParamModel(model, "ExtraTrees", "max_features", np.divide(np.arange(1,9),8), folds, figPath, times)
		#testParamModel(model, "ExtraTrees", "n_estimators", [5, 10, 20, 30, 50, 100], folds, figPath, times)
		testParamModel(model, "ExtraTrees", "max_depth", [10, 20, 30, 50], folds, figPath, times)

	elif modelName == "ADABoost":
		
		#base = getRegModel("MLP")
		base = DecisionTreeRegressor(random_state = 165391, max_depth = 50)
		# define ensemble model
		model = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 5)

		#testParamModel(model, "ADABoost", "n_estimators", [5, 10, 20, 50], folds, figPath, times)
		testParamModel(model, "ADABoost", "max_depth", [10, 20, 30, 50], folds, figPath, times, is_estimator_param	= True)

	elif modelName == "Bagging":
		testParamModel(model, "ExtraTrees", "n_estimators", [100], folds, figPath, times)

	elif modelName == "DummyMean":
		testParamModel(model, "DummyMean", "n_estimators", [100], folds, figPath, times)
	
	elif modelName == "DummyMedian":
		print("More test not available for "+modelName)

	elif modelName == "ADABoost_new":
		
		#base = getRegModel("MLP")
		#base = DecisionTreeRegressor(random_state = 165391)
		model = AdaBoostRegressor(random_state = 165391)#, base_estimator=base)

		testParamModel(model, "ADABoost", "n_estimators", [5, 10, 20, 50], folds, figPath, times)
		#testParamModel(model, "ADABoost", "max_depth", [10, 20, 30, 40, 50], folds, figPath, times, is_estimator_param	= True)



if __name__ == "__main__":
	

	datasetPath = "datasets/mydt/folds"
	
	figPath = "plots/"

	make_plt = False
	modelNames = []
	doMoreTest = True

	#Load the dataset and the feature to compute the x.nnz

	
	featureIndicesL = np.array([0,1,2,3,20,21,22,23])
	featureIndicesU = np.array([0,5,6,7,24,25,26,27])

	pred_indices_L = np.array([4])
	pred_indices_U = np.array([8])

	if not os.path.isdir(figPath+"/NNZXReg"):
		os.mkdir(figPath+"/NNZXReg")




	featureNames = ["solve-type", "L/U size", "L/U NNZ","b NNZ", "f0", "f1", "f2", "f3"]

	list_folds, list_fold_times = load_folds(datasetPath, featureIndicesL, featureIndicesU, pred_indices_L, pred_indices_U, return_times = True)


	regressorNames = ["LinearRegression", "DecisionTree", "MLP", "KNN",
		"RandomForest", "ExtraTrees", "ADABoost"]
	trainRegressors = np.ones(len(regressorNames))
	
	
	for i in range(len(regressorNames)):

		if trainRegressors[i]:
			
			reg = getRegModel(regressorNames[i])
			print("---------------------------------------------"+regressorNames[i]+"--------------------------------------------\n")
			print("Starting the training and testing of model "+regressorNames[i]+" to predict the number of non-zero elements" )
			

			testParams(regressorNames[i], reg, list_folds, figPath, list_fold_times)

			
				
