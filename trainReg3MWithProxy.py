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
# This file will train the 3 regressions for the prediction of the
#	3 algorithm times
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




def getParamModelFromNb(model_nb):

	if model_nb == 0:
		return LinearRegression(), None, None, "LR"

	if model_nb == 1:

		param_name = "max_depth"
		param_values = [10, 20, 30, 40, 50, 80]
		
		return DecisionTreeRegressor(random_state = 165391), param_name, param_values, "DT"


	if model_nb == 2:
		param_name = "max_depth"
		param_values = [10, 20, 30, 40, 50, 80]
		return RandomForestRegressor(random_state = 165391, n_estimators = 2), param_name, param_values,"RF"
	
	if model_nb == 3:
		param_name = "max_depth"
		param_values = [10, 20, 30, 40, 50, 80]
		model = ExtraTreesRegressor(random_state = 165391, n_estimators = 2)

		return model, param_name, param_values,"ET"

	if model_nb == 4:

		base = DecisionTreeRegressor(random_state = 165391, max_depth = 50)
		model = AdaBoostRegressor(random_state = 165391, base_estimator = base, n_estimators = 2)
		param_name = "base_estimator__max_depth"
		param_values = [10, 20, 30, 40, 50, 80]
		return model, param_name, param_values,"Ada"
	

def testParms(modelName, regTimeModel, list_folds, time_per_fold, param_name, param_values, index_reg_general, index_reg_2phase, index_reg_1phase, index_reg_proxy, regProxyModel, regProxyModelName, isLR = False):
	
	if param_name is None:
		col_names = ["Algorithm", "$R^2$ test", "$R^2$ train","MAPE test", "MAPE train", "$\\beta_{1,2}$","$\\gamma_{1,2}$"]
		table = np.zeros((3, len(col_names)))
		param_values = [0]
		j = 0
	else:
		col_names = ["Algorithm", param_name, "$R^2$ test", "$R^2$ train","MAPE test", "MAPE train", "$\\beta_{1,2}$","$\\gamma_{1,2}$"]
		table = np.zeros((3*len(param_values), len(col_names)))
		j = 1

	for i in range(len(param_values)):


		if not param_name is None:
			param = {param_name: param_values[i]}
			regTimeModel = regTimeModel.set_params(**param)


		predAlgoModel = R3MwProxy(modelName, regGeneral = regTimeModel, reg2Phase = regTimeModel, reg1Phase = regTimeModel)

		fold_pred_algo, fold_best_algo, scores, average_times_folds, times_metrics = predAlgoModel.pred_time_cv(list_folds, time_per_fold,  [index_reg_general, index_reg_2phase,
									index_reg_1phase, index_reg_proxy], proxy_model = regProxyModel, proxy_model_name = regProxyModelName)




		r2_test = scores[0]
		r2_train = scores[1]
		MAPE_test = scores[2]
		MAPE_train = scores[3]

		index_algo = np.array([i, len(param_values) + i, 2*len(param_values) + i])

		if not param_name is None:
			table[index_algo, j] = param_values[i]

		table[index_algo, j+1] = r2_test
		table[index_algo, j+2] = r2_train
		table[index_algo, j+3] = MAPE_test
		table[index_algo, j+4] = MAPE_train

		time_metric_general = times_metrics[1]
		time_metric_2phase = times_metrics[2]
		time_metric_1phase = times_metrics[3]

		table[index_algo[0], j+5] = time_metric_general[0]
		table[index_algo[0], j+6] = time_metric_general[2]
		table[index_algo[1], j+5] = time_metric_2phase[0]
		table[index_algo[1], j+6] = time_metric_2phase[2]
		table[index_algo[2], j+5] = time_metric_1phase[0]
		table[index_algo[2], j+6] = time_metric_1phase[2]

	if not os.path.isdir("results"):
		os.mkdir("results")
	if not os.path.isdir("results/predTime"):
		os.mkdir("results/predTime")

	if not param_name is None:
		export_tex_table(table, col_names, "results/predTime", modelName+"_"+param_name+".tex", printTable = True)
	else:
		export_tex_table(table, col_names, "results/predTime", modelName+".tex", printTable = True)



if __name__ == "__main__":

	datasetPath = "datasets/mydt/folds"

	model_names = []



	featureIndicesL = np.array([0,1,2,3,20,21,22,23],)
	featureIndicesU = np.array([0,5,6,7,24,25,26,27])

	yIndicesL = np.array([11, 12, 13, 4])
	yIndicesU = np.array([14, 15, 16, 8])


	list_folds, time_per_fold = load_folds(datasetPath, featureIndicesL, featureIndicesU, yIndicesL, yIndicesU, return_times = True)


	regProxyModelNames = ["LR_1","DT_1","RF_1","Ada_1"]

	for model_nb in range(5):
		
		regTimeModel, param_name, param_values, model_name  = getParamModelFromNb(model_nb)
		

		for proxyNb in range(len(regProxyModelNames)):
			
			regTimeModel = clone(regTimeModel)

			index_reg_general = np.arange(9)
			index_reg_2phase = np.arange(9)
			index_reg_1phase = np.arange(9)
			index_reg_proxy = np.arange(8)
		
			regProxyModel = getProxyModelFromName(regProxyModelNames[proxyNb])

			model_name = model_name + "_" + regProxyModelNames[proxyNb]

			testParms(model_name, regTimeModel, list_folds, time_per_fold, param_name, param_values, index_reg_general, index_reg_2phase,
										index_reg_1phase, index_reg_proxy, regProxyModel, regProxyModelNames[proxyNb])

