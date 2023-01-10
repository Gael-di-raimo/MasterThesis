import numpy as np
import os
import time
import matplotlib.pyplot as plt
import random

from datetime import date

from sklearn.linear_model import SGDRegressor, LassoLars, LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def plot_lines_mean2(list_x, list_y, xlabel = None, ylabel = None, labels = None, nb_points = 100, fig_path = None,
	min_val = None, max_val = None, use_median = False, plot_bar = False, list_colors = None, bar_multiplier = 1):

	plt.clf()
	
	for j in range(len(list_x)):

		x = np.array(list_x[j])
		y = np.array(list_y[j])
		
		if min_val != None and max_val != None:
			mask = np.logical_and(x <= max_val, x >= min_val)
			x = x[mask]
			y = y[mask]

		if len(x) == 0 or len(y) == 0:
			print("Had a len(x) = 0 or len(y) == 0 with a plot for fig "+str(fig_path))
			return

		step = (np.max(x) - np.min(x)) /nb_points
		new_x = []
		new_y = []

		if plot_bar:
			err_y = []

		for i in range(nb_points):
			
			mask = np.logical_and(x < (i+1)*step, x >= i*step)
			if i == nb_points-1:
				mask = np.logical_and(x <= (i+1)*step, x >= i*step)

			masked_x = x[mask]

			if len(masked_x)!= 0 and not use_median: 
				new_x.append(np.mean(x[mask]))
				new_y.append(np.mean(y[mask]))

			if len(masked_x)!= 0 and use_median:
				new_x.append(np.median(x[mask]))
				new_y.append(np.median(y[mask]))

				if plot_bar:
					err_y.append(np.std(y[mask])*bar_multiplier)

		if plot_bar:
			plt.errorbar(new_x, new_y, yerr=err_y, fmt='.k', ecolor =  list_colors[j])

		if labels != None:
			if list_colors == None:
				plt.plot(new_x, new_y, label = labels[j])
			else:
				plt.plot(new_x, new_y, list_colors[j], label = labels[j])

		else:
			if list_colors == None:
				plt.plot(new_x, new_y)
			else:
				plt.plot(new_x, new_y, list_colors[j])
			

	if xlabel != None:
		plt.xlabel(xlabel)

	if ylabel != None:
		plt.ylabel(ylabel)

	plt.legend()

	if fig_path != None:
		plt.savefig(fig_path, bbox_inches='tight')
	else:
		plt.show()

def plot_scatter(list_x, list_y, list_colors, xlabel, ylabel, labels, path_fig, plot_individually = False):

	plt.clf()

	path_fig_splitted = path_fig.split('.')

	path_fig = path_fig_splitted[0]

	if plot_individually:

		for i in range(len(list_x)):
			plt.scatter(list_x[i], list_y[i], c = list_colors[i], label = labels[i])
		
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.legend()
			plt.savefig(path_fig+"_"+ labels[i]+".png", dpi= 500, bbox_inches="tight")
			plt.clf()

	for i in range(len(list_x)):

		plt.scatter(list_x[i], list_y[i], c = list_colors[i], label = labels[i])
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.legend()
		plt.savefig(path_fig+".png", dpi= 500, bbox_inches="tight")


def load_dt(path):

	f_timers = open(path+"/timers.txt")
	f_features = open(path+"/features.txt")

	timer_lines = f_timers.readlines()
	feature_lines = f_features.readlines()

	dt = np.zeros((len(feature_lines), 8))


	for i in range(len(timer_lines)):

		timer_line = timer_lines[i].split()
		feature_line = feature_lines[i].split()
		
		frequency = float(timer_line[4])/10000000

		dt[i, 0] = int(feature_line[5]) #n_cons
		dt[i, 1] = int(feature_line[6]) #nb nz matrix
		dt[i, 2] = int(feature_line[7]) #nb nz rhs
		dt[i, 3] = int(feature_line[8]) #nb nz sol
		dt[i, 4] = int(feature_line[17]) # f_0 but with sol


		dt[i, 5:8] = np.array(timer_line[5:8]).astype(float)/frequency#time in 100ns

	return dt


if __name__ == "__main__":

	
	dataset_path = "datasets"

	method_names = ["general","2phases","1phase"]

	#plot the time in fct of the nz of rhs for all instances groupped
	if not os.path.isdir("plots"):
		os.mkdir("plots")

	L_mult_b_path = 'data/L_multiple_b'
	L_mult_x_path = 'data/L_multiple_x'
	x_mult_L_path = 'data/x_multiple_L'
	mult_L_mult_x_path = 'data/mutliple_L_multiple_x'
	mult_L_mult_b_path = 'data/mutliple_L_multiple_b'

	plt_first_dt = False
	plt_first_b_dt = False
	plt_second_dt = False
	plt_third_dt = True
	plt_fourth_dt = False


	plot_hist = False

	if plot_hist:

		dt = load_dt(x_mult_L_path)
		x = dt[:,4]
		plt.hist(x, bins = 10)
		plt.show()
		exit(0)

	if plt_first_dt:
	
		print("Loading the dataset to make plots ... ")
		dt = load_dt(L_mult_b_path)

		print("Loading finished start making the plot.")

		y = dt[:, 5:8] #times
		
		color_methods = ['red','green', 'blue']

		nb_cons = dt[:,0]
		nb_nz_mtx = dt[:,1]
		nb_nz_rhs = dt[:,2]
		nb_nz_sol = dt[:,3]

		log_nb_nz_sol = np.log10(nb_nz_sol)
		log_nb_nz_rhs = np.log10(nb_nz_rhs)
		log_y =  np.log10(y)


		
		#plt time as a function of the number of non-zeros of the rhs and the one of the sol
		nz_rhs_per_method = []
		nz_sol_per_method = []
		log_nz_rhs_per_method = []
		log_nz_sol_per_method = []

		log_cpu_cycles_per_method = []
		cpu_cycles_per_method = []

		for i in range(y.shape[1]):

			nz_rhs_per_method.append(nb_nz_rhs)
			nz_sol_per_method.append(nb_nz_sol)

			log_nz_rhs_per_method.append(log_nb_nz_rhs)
			log_nz_sol_per_method.append(log_nb_nz_sol)


			log_cpu_cycles_per_method.append(log_y[:,i])
			cpu_cycles_per_method.append(y[:,i])

			
		index_gen_1phase = [0,2]
		#plt time as a function of the number of non-zeros of the rhs

		ylabel = "Time taken per 100ns"
		xlabel = "Log10(number of non-zero elements in the right-hand side)"

		fig_path = "plots/first/times_with_log_nz_rhs_dt1.png"
		
		plot_lines_mean2(log_nz_rhs_per_method, cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True, list_colors = color_methods, plot_bar = True)
		
		fig_path = "plots/first/times_with_log_nz_rhs_dt1_scatter.png"

		plot_scatter(log_nz_rhs_per_method, cpu_cycles_per_method, color_methods, xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)



		ylabel = "Time taken per 100ns"
		xlabel = "Log10(number of non-zero elements in the right-hand side)"

		fig_path = "plots/first/times_with_log_nz_rhs_dt1_no_2phase.png"
		
		plot_lines_mean2([log_nz_rhs_per_method[i] for i in index_gen_1phase], [cpu_cycles_per_method[i] for i in index_gen_1phase], xlabel = xlabel
			, ylabel = ylabel, labels = [method_names[i] for i in index_gen_1phase], nb_points = 50, fig_path = fig_path, use_median = True, plot_bar = True, list_colors = [color_methods[i] for i in index_gen_1phase])

		fig_path = "plots/first/times_with_log_nz_rhs_dt1_no_2phase_scatter.png"

		plot_scatter([log_nz_rhs_per_method[i] for i in index_gen_1phase], [cpu_cycles_per_method[i] for i in index_gen_1phase], [color_methods[i] for i in index_gen_1phase], xlabel, ylabel,
			[method_names[i] for i in index_gen_1phase], fig_path)


		


		#plt time as a function of the number of non-zeros of the sol

		ylabel = "Time taken per 100ns"
		xlabel = "Number of non-zero elements in the solution"

		fig_path = "plots/first/times_with_log_nz_sol_dt1.png"
		
		plot_lines_mean2(log_nz_sol_per_method, cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True)
		
		fig_path = "plots/first/times_with_log_nz_sol_dt1_scatter.png"

		plot_scatter(log_nz_sol_per_method, cpu_cycles_per_method, ['red','green', 'blue'], xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)


		ylabel = "Log10(Time taken per 100ns)"
		xlabel = "Log10(Number of non-zero elements in the solution)"

		fig_path = "plots/first/log_times_with_log_nz_sol_dt1.png"
		
		plot_lines_mean2(log_nz_sol_per_method, log_cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True, plot_bar = True, list_colors = color_methods, bar_multiplier = 3)
		
		fig_path = "plots/first/log_times_with_log_nz_sol_dt1_scatter.png"

		plot_scatter(log_nz_sol_per_method, log_cpu_cycles_per_method, ['red','green', 'blue'], xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)





		ylabel = "Time taken per 100ns"
		xlabel = "Log10(number of non-zero elements in the solution)"

		fig_path = "plots/first/times_with_log_nz_sol_dt1_no_2phase.png"
		
		plot_lines_mean2([log_nz_sol_per_method[i] for i in index_gen_1phase], [cpu_cycles_per_method[i] for i in index_gen_1phase], xlabel = xlabel
			, ylabel = ylabel, labels = [method_names[i] for i in index_gen_1phase], nb_points = 50, fig_path = fig_path, use_median = True, plot_bar = True, list_colors = [color_methods[i] for i in index_gen_1phase])

		fig_path = "plots/first/times_with_log_nz_sol_dt1_no_2phase_scatter.png"

		plot_scatter([log_nz_sol_per_method[i] for i in index_gen_1phase], [cpu_cycles_per_method[i] for i in index_gen_1phase], [color_methods[i] for i in index_gen_1phase], xlabel, ylabel,
			[method_names[i] for i in index_gen_1phase], fig_path)

	if plt_first_b_dt:
	
		print("Loading the dataset to make plots ... ")
		dt = load_dt(L_mult_x_path)

		print("Loading finished start making the plot.")

		y = dt[:, 5:8] #times
		
		nb_cons = dt[:,0]
		nb_nz_mtx = dt[:,1]
		nb_nz_rhs = dt[:,2]
		nb_nz_sol = dt[:,3]
		
		print(np.min(nb_nz_sol))
		#plt time as a function of the number of non-zeros of the rhs and the one of the sol
		nz_rhs_per_method = []
		nz_sol_per_method = []
		cpu_cycles_per_method = []

		for i in range(y.shape[1]):

			nz_rhs_per_method.append(nb_nz_rhs)
			nz_sol_per_method.append(nb_nz_sol)
			cpu_cycles_per_method.append(y[:,i])

			

		#plt time as a function of the number of non-zeros of the rhs

		ylabel = "Time taken per 100ns"
		xlabel = "Number of non-zero elements in the right-hand side"

		fig_path = "plots/first_b/times_with_nz_rhs_dt1.png"
		
		plot_lines_mean2(nz_rhs_per_method, cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True)
		
		fig_path = "plots/first_b/times_with_nz_rhs_dt1_scatter.png"

		plot_scatter(nz_rhs_per_method, cpu_cycles_per_method, ['red','green', 'blue'], xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)



		#plt time as a function of the number of non-zeros of the sol

		ylabel = "Time taken per 100ns"
		xlabel = "Number of non-zero elements in the solution"

		fig_path = "plots/first_b/times_with_nz_sol_dt1.png"
		
		plot_lines_mean2(nz_sol_per_method, cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True)
		
		fig_path = "plots/first_b/times_with_nz_sol_dt1_scatter.png"

		plot_scatter(nz_sol_per_method, cpu_cycles_per_method, ['red','green', 'blue'], xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)
	
	if plt_second_dt:

		y = dt[:, 5:8] #times
		
		nb_cons = dt[:,0]
		nb_nz_mtx = dt[:,1]
		nb_nz_rhs = dt[:,2]
		nb_nz_sol = dt[:,3]

		sum_corres_xnz_L =  dt[:,4]
		
		#plt time as a function of the sum of the number of non-zeros in the corresponding col where x is nz in L
		nz_L_per_method = []
		nz_sol_per_method = []
		cpu_cycles_per_method = []

		for i in range(y.shape[1]):

			nz_L_per_method.append(sum_corres_xnz_L)
			cpu_cycles_per_method.append(y[:,i])

			

		#plt

		ylabel = "Time taken per 100ns"
		xlabel = "Number of non-zero elements in column of L corresponding to a non-zero of x"

		fig_path = "plots/second/times_with_nz_L_dt2.png"
		
		plot_lines_mean2(nz_L_per_method, cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True)
		
		fig_path = "plots/second/times_with_nz_L_dt2_scatter.png"

		plot_scatter(nz_L_per_method, cpu_cycles_per_method, ['red','green', 'blue'], xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)

	if plt_third_dt:

		print("Loading the dataset to make plots ... ")
		dt = load_dt(mult_L_mult_x_path)

		print("Loading finished start making the plot.")

		sub_dt = []
		i = 0

		dt[:,4] = np.divide(dt[:,4],dt[:,3]).astype(int)

		while len(dt) != 0:
			mask = dt[:,4] == dt[0,4]
			
			sub_dt.append(dt[mask,:])
			dt = dt[np.logical_not(mask),:]

			i += 1

		
		for i in range(len(method_names)):

			legend = []
			nz_L_per_method = []
			nz_sol_per_method = []
			cpu_cycles_per_method = []
			sum_corres_xnz_L = []


			nb_lines = np.zeros(len(sub_dt))

			j = 0
			for data in sub_dt:
				nb_lines[j] = data.shape[0]
				j = j + 1
			
			mean_nb_lines =np.mean(nb_lines)

			for data in sub_dt:
				#general algorithm
				if data.shape[0] > mean_nb_lines:
					y = data[:, 5+i] #times
					
					nb_cons = data[:,0]
					nb_nz_mtx = data[:,1]
					nb_nz_rhs = data[:,2]
					nb_nz_sol = data[:,3]

					#make the legend
					legend.append(r'$\alpha$/x.nz = '+str(data[0,4]))

					nz_sol_per_method.append(nb_nz_sol)
					cpu_cycles_per_method.append(y)

			ylabel = "Time taken per 100ns"
			xlabel = "Number of non-zero elements in the solution"

			fig_path = "plots/third/times_with_nz_x_dt3_"+method_names[i]+".png"
			
			plot_lines_mean2(nz_sol_per_method, cpu_cycles_per_method, xlabel = xlabel
				, ylabel = ylabel, labels = legend, nb_points = 50, fig_path = fig_path, use_median = True)

		"""
		y = dt[:, 5:8] #times
		
		nb_cons = dt[:,0]
		nb_nz_mtx = dt[:,1]
		nb_nz_rhs = dt[:,2]
		nb_nz_sol = dt[:,3]

		sum_corres_xnz_L =  dt[:,4]
		
		
		#plt time as a function of the sum of the number of non-zeros in the corresponding col where x is nz in L
		nz_L_per_method = []
		nz_sol_per_method = []
		cpu_cycles_per_method = []

		for i in range(y.shape[1]):

			nz_sol_per_method.append(nb_nz_sol)
			nz_L_per_method.append(sum_corres_xnz_L)
			cpu_cycles_per_method.append(y[:,i])

			

		#plt

		ylabel = "Time taken per 100ns"
		xlabel = "Number of non-zero elements in column of L corresponding to a non-zero of x"

		fig_path = "plots/third/times_with_nz_L_dt3.png"
		
		plot_lines_mean2(nz_L_per_method, cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True)
		
		fig_path = "plots/third/times_with_nz_L_dt3_scatter.png"

		plot_scatter(nz_L_per_method, cpu_cycles_per_method, ['red','green', 'blue'], xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)


		ylabel = "Time taken per 100ns"
		xlabel = "Number of non-zero elements in x"

		fig_path = "plots/third/nz_sol_per_method.png"
		
		plot_lines_mean2(nz_sol_per_method, cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True)
		
		fig_path = "plots/third/nz_sol_per_method.png"

		plot_scatter(nz_sol_per_method, cpu_cycles_per_method, ['red','green', 'blue'], xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)
		"""

	if plt_fourth_dt:
	
		print("Loading the dataset to make plots ... ")
		dt = load_dt(mult_L_mult_b_path)

		print("Loading finished start making the plot.")

		y = dt[:, 5:8] #times
		
		color_methods = ['red','green', 'blue']

		nb_cons = dt[:,0]
		nb_nz_mtx = dt[:,1]
		nb_nz_rhs = dt[:,2]
		nb_nz_sol = dt[:,3]

		log_nb_nz_sol = np.log10(nb_nz_sol)
		log_nb_nz_rhs = np.log10(nb_nz_rhs)
		log_y =  np.log10(y)


		
		#plt time as a function of the number of non-zeros of the rhs and the one of the sol
		nz_rhs_per_method = []
		nz_sol_per_method = []
		log_nz_rhs_per_method = []
		log_nz_sol_per_method = []

		log_cpu_cycles_per_method = []
		cpu_cycles_per_method = []

		for i in range(y.shape[1]):

			nz_rhs_per_method.append(nb_nz_rhs)
			nz_sol_per_method.append(nb_nz_sol)

			log_nz_rhs_per_method.append(log_nb_nz_rhs)
			log_nz_sol_per_method.append(log_nb_nz_sol)


			log_cpu_cycles_per_method.append(log_y[:,i])
			cpu_cycles_per_method.append(y[:,i])

			
		index_gen_1phase = [0,2]
		#plt time as a function of the number of non-zeros of the rhs

		ylabel = "Time taken per 100ns"
		xlabel = "Log10(number of non-zero elements in the right-hand side)"

		fig_path = "plots/fourth/times_with_log_nz_rhs_dt4.png"
		
		plot_lines_mean2(log_nz_rhs_per_method, cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True, list_colors = color_methods, plot_bar = True)
		
		fig_path = "plots/fourth/times_with_log_nz_rhs_dt4_scatter.png"

		plot_scatter(log_nz_rhs_per_method, cpu_cycles_per_method, color_methods, xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)



		ylabel = "Time taken per 100ns"
		xlabel = "Log10(number of non-zero elements in the right-hand side)"

		fig_path = "plots/fourth/times_with_log_nz_rhs_dt4_no_2phase.png"
		
		plot_lines_mean2([log_nz_rhs_per_method[i] for i in index_gen_1phase], [cpu_cycles_per_method[i] for i in index_gen_1phase], xlabel = xlabel
			, ylabel = ylabel, labels = [method_names[i] for i in index_gen_1phase], nb_points = 50, fig_path = fig_path, use_median = True, plot_bar = True, list_colors = [color_methods[i] for i in index_gen_1phase])

		fig_path = "plots/fourth/times_with_log_nz_rhs_dt4_no_2phase_scatter.png"

		plot_scatter([log_nz_rhs_per_method[i] for i in index_gen_1phase], [cpu_cycles_per_method[i] for i in index_gen_1phase], [color_methods[i] for i in index_gen_1phase], xlabel, ylabel,
			[method_names[i] for i in index_gen_1phase], fig_path)


		


		#plt time as a function of the number of non-zeros of the sol

		ylabel = "Time taken per 100ns"
		xlabel = "Number of non-zero elements in the solution"

		fig_path = "plots/fourth/times_with_log_nz_sol_dt4.png"
		
		plot_lines_mean2(log_nz_sol_per_method, cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True)
		
		fig_path = "plots/fourth/times_with_log_nz_sol_dt4_scatter.png"

		plot_scatter(log_nz_sol_per_method, cpu_cycles_per_method, ['red','green', 'blue'], xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)


		ylabel = "Log10(Time taken per 100ns)"
		xlabel = "Log10(Number of non-zero elements in the solution)"

		fig_path = "plots/fourth/log_times_with_log_nz_sol_dt4.png"
		
		plot_lines_mean2(log_nz_sol_per_method, log_cpu_cycles_per_method, xlabel = xlabel
			, ylabel = ylabel, labels = method_names[0:3], nb_points = 50, fig_path = fig_path, use_median = True, plot_bar = True, list_colors = color_methods, bar_multiplier = 3)
		
		fig_path = "plots/fourth/log_times_with_log_nz_sol_dt4_scatter.png"

		plot_scatter(log_nz_sol_per_method, log_cpu_cycles_per_method, ['red','green', 'blue'], xlabel, ylabel, method_names[0:3], fig_path, plot_individually = True)





		ylabel = "Time taken per 100ns"
		xlabel = "Log10(number of non-zero elements in the solution)"

		fig_path = "plots/fourth/times_with_log_nz_sol_dt4_no_2phase.png"
		
		plot_lines_mean2([log_nz_sol_per_method[i] for i in index_gen_1phase], [cpu_cycles_per_method[i] for i in index_gen_1phase], xlabel = xlabel
			, ylabel = ylabel, labels = [method_names[i] for i in index_gen_1phase], nb_points = 50, fig_path = fig_path, use_median = True, plot_bar = True, list_colors = [color_methods[i] for i in index_gen_1phase])

		fig_path = "plots/fourth/times_with_log_nz_sol_dt4_no_2phase_scatter.png"

		plot_scatter([log_nz_sol_per_method[i] for i in index_gen_1phase], [cpu_cycles_per_method[i] for i in index_gen_1phase], [color_methods[i] for i in index_gen_1phase], xlabel, ylabel,
			[method_names[i] for i in index_gen_1phase], fig_path)



	

	

