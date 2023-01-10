import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random


from utility.utility_plot import  plot_per_b_size, plotRegressions, plotIntersections, plot_per_b_size_folds
from utility.load_dt import load_dt_model
from utility.my_k_fold_cv import load_folds, my_cv_score
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



# --------------------------------------------------------------------
# This file trains time regressors selected to see the time taken in
# 	average in the test folds
# --------------------------------------------------------------------


def getProxyModelFromName(modelName):

	if modelName == "LR_1":
		return LinearRegression()

	if modelName == "DT_1":
		return DecisionTreeRegressor(random_state = 165391, max_depth = 50)

	if modelName == "RF_1":
		return RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 20)
	
	if modelName == "Ada_1":

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 20)
		model = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 5)

		return model


if __name__ == "__main__":

	datasetPath = "datasets/mydt/folds"

	model_names = []



	featureIndicesL = np.array([0,1,2,3,20,21,22,23],)
	featureIndicesU = np.array([0,5,6,7,24,25,26,27])

	yIndicesL = np.array([11, 12, 13, 4])
	yIndicesU = np.array([14, 15, 16, 8])


	list_folds, time_per_fold = load_folds(datasetPath, featureIndicesL, featureIndicesU, yIndicesL, yIndicesU, return_times = True)

		
	testBestDT = True
	testBestRF = True
	testBestET = True
	testBestAda = True



	if not os.path.isdir("results"):
		os.mkdir("results")

	if not os.path.isdir("results/classifier"):
		os.mkdir("results/classifier")

	if testBestDT:
		regProxyModelNames = ["LR_1","DT_1","RF_1","Ada_1"]


		nb_model = 5
		table = np.zeros((nb_model,6))
		col_names = ["Model","Average time (s)", "Balanced accuracy", "$\\beta_{1,2}$","$\\gamma_{1,2}$","$\\gamma_{1,3}$"]

		
		#Model 1

		model_nb = 0
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 0
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "DTLRProxy"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = DecisionTreeRegressor(random_state = 165391, max_depth = 30),
								reg1Phase =  DecisionTreeRegressor(random_state = 165391, max_depth = 20))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]




		#Model 2
		model_nb = 1
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 1
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "DTDTProxy"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = DecisionTreeRegressor(random_state = 165391, max_depth = 20),
								reg1Phase =  DecisionTreeRegressor(random_state = 165391, max_depth = 20))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]




		#Model 3
		model_nb = 2
		proxyNb = 2
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])

		classifier_name = "DTRFProxy"
		LRNNZBb = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = DecisionTreeRegressor(random_state = 165391, max_depth = 20),
								reg1Phase =  DecisionTreeRegressor(random_state = 165391, max_depth = 20))

		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)




		#Model 4

		model_nb = 3
		proxyNb = 3
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "DTAdaProxy"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = DecisionTreeRegressor(random_state = 165391, max_depth = 30),
								reg1Phase =  DecisionTreeRegressor(random_state = 165391, max_depth = 20))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

				

		tot_time_metrics = times_metrics[0]


		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		#Model 5
		model_nb = 4
		LRNNZBb = R3MwProxy("LRNNZBb")

		index_reg_general = np.array([1])
		index_reg_2phase =  np.array([3])
		index_reg_1phase =  np.array([3])
		index_reg_proxy = np.arange(8)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = LRNNZBb.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = None, proxy_model_name = None)
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		export_tex_table(table, col_names, 'results/classifier',"LRNNZBb_"+classifier_name+'.tex', printTable = True)

	if testBestRF:
		regProxyModelNames = ["LR_1","DT_1","RF_1","Ada_1"]


		nb_model = 8
		table = np.zeros((nb_model,6))
		col_names = ["Model","Average time (s)", "Balanced accuracy", "$\\beta_{1,2}$","$\\gamma_{1,2}$","$\\gamma_{1,3}$"]

		
		#Model 1

		model_nb = 0
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 0
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "RFLRProxy"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 40),
								reg1Phase =  RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 50))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]




		#Model 2
		model_nb = 1
		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 1
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "RFDTProxy (a)"

		classifier = R3MwProxy(classifier_name, regGeneral = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 10), reg2Phase = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 40),
								reg1Phase =  RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 40))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]
		
		#Model 3

		model_nb = model_nb + 1
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 1
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "RFDTProxy (b)"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 40),
								reg1Phase =  RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 40))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]




		#Model 4
		model_nb = model_nb + 1
		proxyNb = 2
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])

		classifier_name = "RFRFProxy (a)"
		LRNNZBb = R3MwProxy(classifier_name, regGeneral = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 10), reg2Phase = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 30),
								reg1Phase =  RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 20))

		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		#Model 5
		model_nb = model_nb + 1
		proxyNb = 2
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])

		classifier_name = "RFRFProxy (b)"
		LRNNZBb = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 30),
								reg1Phase =  RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 20))

		index_reg_general =  np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		



		#Model 6

		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		model_nb = model_nb+1
		proxyNb = 3
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "RFAdaProxy (a)"

		classifier = R3MwProxy(classifier_name, regGeneral = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 10), reg2Phase = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 20),
								reg1Phase =  RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 20))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

				

		tot_time_metrics = times_metrics[0]
		tot_time_metrics = times_metrics[0]


		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]
		
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		#Model 7
		model_nb = model_nb+1
		proxyNb = 3
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "RFAdaProxy (b)"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 20),
								reg1Phase =  RandomForestRegressor(random_state = 165391, n_estimators = 2, max_depth = 20))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

				

		tot_time_metrics = times_metrics[0]
		tot_time_metrics = times_metrics[0]


		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]
		
		#Model 8
		model_nb = model_nb+1
		LRNNZBb = R3MwProxy("LRNNZBb")

		index_reg_general = np.array([1])
		index_reg_2phase =  np.array([3])
		index_reg_1phase =  np.array([3])
		index_reg_proxy = np.arange(8)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = LRNNZBb.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = None, proxy_model_name = None)
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		export_tex_table(table, col_names, 'results/classifier',"LRNNZBb_AllProxyWithRf.tex", printTable = True)

	if testBestET:
		regProxyModelNames = ["LR_1","DT_1","RF_1","Ada_1"]


		nb_model = 9
		table = np.zeros((nb_model,6))
		col_names = ["Model","Average time (s)", "Balanced accuracy", "$\\beta_{1,2}$","$\\gamma_{1,2}$","$\\gamma_{1,3}$"]

		
		#Model 1

		model_nb = 0
		
		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 0
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "ETLRProxy (a)"

		classifier = R3MwProxy(classifier_name, regGeneral = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 10),
								reg2Phase = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 80),
								reg1Phase =  ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 30))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		#Model 2
		model_nb = model_nb + 1
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 0
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "ETLRProxy (b)"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(),
								reg2Phase = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 80),
								reg1Phase =  ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 30))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]


		#Model 3
		model_nb = model_nb + 1
		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 1
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "ETDTProxy (a)"

		classifier = R3MwProxy(classifier_name, regGeneral = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 10),
								reg2Phase = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 80),
								reg1Phase =  ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 30))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]
		
		#Model 4

		model_nb = model_nb + 1
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 1
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "ETDTProxy (b)"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), 
								reg2Phase = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 80),
								reg1Phase =  ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 30))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]




		#Model 5
		model_nb = model_nb + 1
		proxyNb = 2
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])

		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		classifier_name = "ETRFProxy (a)"
		LRNNZBb = R3MwProxy(classifier_name, regGeneral = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 10),
								 reg2Phase = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 50),
								reg1Phase =  ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 40))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		#Model 6
		model_nb = model_nb + 1
		proxyNb = 2
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])

		classifier_name = "ETRFProxy (b)"
		LRNNZBb = R3MwProxy(classifier_name, regGeneral = LinearRegression(),
								reg2Phase = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 50),
								reg1Phase =  ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 40))

		index_reg_general =  np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		



		#Model 7

		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		model_nb = model_nb+1
		proxyNb = 3
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "ETAdaProxy (a)"

		classifier = R3MwProxy(classifier_name, regGeneral = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 20),
								reg2Phase = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 30),
								reg1Phase =  ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 30))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

				

		tot_time_metrics = times_metrics[0]
		tot_time_metrics = times_metrics[0]


		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]
		
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		#Model 8
		model_nb = model_nb+1
		proxyNb = 3
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "ETAdaProxy (b)"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(),
								reg2Phase = ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 30),
								reg1Phase =  ExtraTreesRegressor(random_state = 165391, n_estimators = 2, max_depth = 30))

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

				

		tot_time_metrics = times_metrics[0]
		tot_time_metrics = times_metrics[0]


		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]
		
		#Model 9
		model_nb = model_nb+1
		LRNNZBb = R3MwProxy("LRNNZBb")

		index_reg_general = np.array([1])
		index_reg_2phase =  np.array([3])
		index_reg_1phase =  np.array([3])
		index_reg_proxy = np.arange(8)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = LRNNZBb.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = None, proxy_model_name = None)
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		export_tex_table(table, col_names, 'results/classifier',"LRNNZBb_AllProxyWithET.tex", printTable = True)

	if testBestAda:
		regProxyModelNames = ["LR_1","DT_1","RF_1","Ada_1"]


		nb_model = 9
		table = np.zeros((nb_model,6))
		col_names = ["Model","Average time (s)", "Balanced accuracy", "$\\beta_{1,2}$","$\\gamma_{1,2}$","$\\gamma_{1,3}$"]

		
		#Model 1

		model_nb = 0
		
		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 0
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "AdaLRProxy (a)"

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 10)
		regGeneral = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 20)
		reg2Phase = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 20)
		reg1Phase = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		classifier = R3MwProxy(classifier_name, regGeneral = regGeneral, reg2Phase = reg2Phase, reg1Phase = reg1Phase)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		#Model 2
		model_nb = model_nb + 1
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 0
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "AdaLRProxy (b)"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = reg2Phase, reg1Phase =  reg1Phase)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]


		#Model 3
		model_nb = model_nb + 1
		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 1
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "AdaDTProxy (a)"

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 10)
		regGeneral = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 30)
		reg2Phase = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 20)
		reg1Phase = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		classifier = R3MwProxy(classifier_name, regGeneral = regGeneral, reg2Phase = reg2Phase, reg1Phase = reg1Phase)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]
		
		#Model 4

		model_nb = model_nb + 1
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		proxyNb = 1
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "AdaDTProxy (b)"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = reg2Phase, reg1Phase = reg1Phase)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

		tot_time_metrics = times_metrics[0]

		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]




		#Model 5
		model_nb = model_nb + 1
		proxyNb = 2
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])

		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		classifier_name = "AdaRFProxy (a)"

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 10)
		regGeneral = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 30)
		reg2Phase = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 20)
		reg1Phase = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		classifier = R3MwProxy(classifier_name, regGeneral = regGeneral, reg2Phase = reg2Phase, reg1Phase = reg1Phase)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		#Model 6
		model_nb = model_nb + 1
		proxyNb = 2
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])

		index_reg_general =  np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		classifier_name = "AdaRFProxy (b)"
		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = reg2Phase, reg1Phase = reg1Phase)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		



		#Model 7

		index_reg_general = np.arange(9)
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		model_nb = model_nb+1
		proxyNb = 3
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])
		classifier_name = "AdaAdaProxy (a)"

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 10)
		regGeneral = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 20)
		reg2Phase = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 20)
		reg1Phase = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)

		classifier = R3MwProxy(classifier_name, regGeneral = regGeneral, reg2Phase = reg2Phase, reg1Phase = reg1Phase)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

				

		tot_time_metrics = times_metrics[0]
		tot_time_metrics = times_metrics[0]


		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]
		
		index_reg_general = np.array([1])
		index_reg_2phase = np.arange(9)
		index_reg_1phase = np.arange(9)
		index_reg_proxy = np.arange(8)

		#Model 8
		model_nb = model_nb+1
		proxyNb = 3
		regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])

	

		classifier_name = "AdaAdaProxy (b)"

		classifier = R3MwProxy(classifier_name, regGeneral = LinearRegression(), reg2Phase = reg2Phase, reg1Phase = reg1Phase)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = classifier.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelNames[proxyNb])

				

		tot_time_metrics = times_metrics[0]
		tot_time_metrics = times_metrics[0]


		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]
		
		#Model 9
		model_nb = model_nb+1
		LRNNZBb = R3MwProxy("LRNNZBb")

		index_reg_general = np.array([1])
		index_reg_2phase =  np.array([3])
		index_reg_1phase =  np.array([3])
		index_reg_proxy = np.arange(8)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = LRNNZBb.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = None, proxy_model_name = None)
		
		tot_time_metrics = times_metrics[0]

		
		table[model_nb,1] = average_times_folds[3]
		table[model_nb,2] = scores[4]
		table[model_nb,3] = tot_time_metrics[0]
		table[model_nb,4] = tot_time_metrics[2]

		export_tex_table(table, col_names, 'results/classifier',"LRNNZBb_AllProxyWithAda.tex", printTable = True)

