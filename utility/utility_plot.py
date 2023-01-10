import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import mean_absolute_percentage_error

def plot_lines_mean(path, use_mean, nb_pts, splitted_dataset, regressor):


	method_name = ["dense","bitqueue_1","bitqueue_2","bitqueue_3","bitqueue_1b","heap","2phase_radixsort","2phase_quicksort"]
	color = ["g","b","r","y","k","c","m","g-"]
	print("start the plot ")
	
	if seperated_dataset:
		log_path = 	"D:/dataset_proxy_justif/results/metric_method_"+str(method_nb)+"_"+dataset_names[j]+"_"+str(today)+".txt"
	else:
		log_path = 	"D:/dataset_proxy_justif/results/metric_method_"+str(method_nb)+"_"+str(today)+".txt"

	for method_num in range(8):
		
		load_path = path+"/method_"+str(method_num)+"_"+method_name[i]+"/"+pred+"/"+regressor

		x_test = np.loadtxt(load_path+"/x_test.csv", delimiter = ",", dtype = int)
		y_test = np.loadtxt(load_path+"/y_test.csv", delimiter = ",", dtype = int)
		y_pred_test = np.loadtxt(load_path+"/y_pred_test.csv", delimiter = ",", dtype = int)
	

	
		step = np.max(x)/400

		for i in range(400):
			
			mask = ((x < (i+1)*step) and (x >= i*step))
			if i == 399:
				mask = ((x <= (i+1)*step) and (x >= i*step))

			masked_x = x[mask]
			
			if len(masked_x)!= 0: 
				new_x.append(np.mean(x[mask]))
				new_y.append(np.mean(y[mask]))

		print(new_x)
		print(new_y)
		plt.plot(new_x, new_y, color[method_num], label = method_name[method_num])

	plt.legend()
	plt.show()

def plot_lines_mean2(list_x, list_y, xlabel = None, ylabel = None, labels = None, nb_points = 100, fig_path = None, min_val = None, max_val = None, use_median = False):

	plt.clf()
	
	for j in range(len(list_x)):

		x = np.array(list_x[j])
		y = np.array(list_y[j])
		
		if min_val != None and max_val != None:
			mask = np.logical_and(x <= max_val, x >= min_val)
			x = x[mask]
			y = y[mask]

		if len(x) == 0 or len(y) == 0:
			print("Had a len(x) = 0 or len(y) == 0 with a plot for fig %s lenx = %d  leny = %d" %(fig_path, len(x), len(y)))
			return

		step = (np.max(x) - np.min(x)) /nb_points
		print(step)
		new_x = []
		new_y = []

		for i in range(nb_points):
			
			mask = np.logical_and(x < np.min(x) + (i+1)*step, x >= np.min(x) + i*step)
			if i == nb_points-1:
				mask = np.logical_and(x <=  np.min(x) + (i+1)*step, x >=  np.min(x) + i*step)

			masked_x = x[mask]

			if len(masked_x)!= 0 and not use_median: 
				new_x.append(np.mean(x[mask]))
				new_y.append(np.mean(y[mask]))

			if len(masked_x)!= 0 and use_median: 
				new_x.append(np.median(x[mask]))
				new_y.append(np.median(y[mask]))



		if labels != None:
			plt.plot(new_x, new_y, label = labels[j])
		else:
			plt.plot(new_x, new_y)

	if xlabel != None:
		plt.xlabel(xlabel)

	if ylabel != None:
		plt.ylabel(ylabel)

	plt.legend()

	if fig_path != None:
		plt.savefig(fig_path)
	else:
		plt.show()

def plt_hist(x, xlabel = None, ylabel = None):
	n, bins, patches = plt.hist(x=x, bins='auto', color='b', alpha=0.7, rwidth=0.85)
	plt.grid(axis='y', alpha=0.75)
	#plt.xlabel(xlabel)
	#plt.ylabel(ylabel)

	maxfreq = n.max()
	# Set a clean upper y-axis limit.
	plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

	plt.show()

def plot_scatter(list_x, list_y, list_colors, xlabel, ylabel, labels, path_fig, plot_individually = False):

	plt.clf()
	path_fig_splitted = path_fig.split('.')

	path_fig = path_fig_splitted[0]


	for i in range(len(list_x)):
		plt.scatter(list_x[i], list_y[i], c = list_colors[i], label = labels[i])
		
		if plot_individually:
			plt.xlabel(xlabel)
			plt.ylabel(ylabel)
			plt.legend()
			plt.savefig(path_fig+"_"+ labels[i]+".png", dpi= 300, bbox_inches="tight")
			plt.clf()

	if not plot_individually:
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.legend()
		plt.savefig(path_fig+".png", dpi= 300, bbox_inches="tight")

"""
def plot_per_b_size(best_algo, pred_algo, dt_test, model_names, fig_path = "bar_plot", pred_algo_2 = []):

	b_sizes = np.concatenate((dt_test[:, 1], dt_test[:, 5]))
	times =  np.concatenate((dt_test[:, 11:14], dt_test[:, 14:17]))

	nb_bar = 18
	min_b_sizes = np.min(b_sizes)
	max_b_sizes = np.max(b_sizes)

	start_ranges = np.arange(min_b_sizes,max_b_sizes+1, int((max_b_sizes-min_b_sizes)/15))

	list_time_best_algo = []
	list_time_pred_algo = []
	list_time_average_algo = []
	list_time_pred_algo_2 = []
	labels = []
	best_algo_times = []
	pred_algo_times = []

	if len(pred_algo_2) != 0:
		pred_algo_2_times = []

	for method_num in range(3):
		best_algo_times.append([])
		pred_algo_times.append([])

		if len(pred_algo_2) != 0:
			pred_algo_2_times.append([])


	for j in range(len(start_ranges)-1):

		mask_range = np.logical_and(b_sizes < start_ranges[j+1], b_sizes >=  start_ranges[j])

		pred_algo_in_range = pred_algo[mask_range]
		best_algo_in_range = best_algo[mask_range]
		
		if len(pred_algo_2) != 0:
			pred_algo_2_in_range = pred_algo_2[mask_range]


		if len(pred_algo_in_range) != 0:
		
			labels.append(str(round(start_ranges[j]/10000,1))+'-'+str(round(start_ranges[j+1]/10000,1)))
			times_in_range = times[mask_range,:]

			list_time_pred_algo.append(np.sum(times_in_range[np.arange(0,times_in_range.shape[0]), pred_algo_in_range])/times_in_range.shape[0])
			list_time_best_algo.append(np.sum(times_in_range[np.arange(0,times_in_range.shape[0]), best_algo_in_range])/times_in_range.shape[0])
			list_time_average_algo.append(np.mean(np.mean(times_in_range, axis = 1)))

			if len(pred_algo_2) != 0:
				list_time_pred_algo_2.append(np.sum(times_in_range[np.arange(0,times_in_range.shape[0]), pred_algo_2_in_range])/times_in_range.shape[0])


			#making the bar for algo
			values, counts = np.unique(pred_algo_in_range, return_counts=True)
		
			for method_num in range(3):
				if not (method_num in values):
					values = np.append(values, method_num)
					counts = np.append(counts, 0)

			for v in range(len(values)):
				pred_algo_times[values[v]].append(counts[v]*100/np.sum(np.array(counts)))


			#making the bar for the best algo
			values, counts = np.unique(best_algo_in_range, return_counts=True)
		
			for method_num in range(3):
				if not (method_num in values):
					values = np.append(values, method_num)
					counts = np.append(counts, 0)
			
			for v in range(len(values)):
				best_algo_times[values[v]].append(counts[v]*100/np.sum(np.array(counts)))

			if len(pred_algo_2) != 0:
				#making the bar for algo 2
				values, counts = np.unique(pred_algo_2_in_range, return_counts=True)
			
				for method_num in range(3):
					if not (method_num in values):
						values = np.append(values, method_num)
						counts = np.append(counts, 0)

				for v in range(len(values)):
					pred_algo_2_times[values[v]].append(counts[v]*100/np.sum(np.array(counts)))




	x = np.arange(len(list_time_best_algo))*0.9

	plt.clf()


	fig = plt.subplots(figsize =(22, 8))
	width = 0.2
	tick_font_size = 18

	plt.bar(x, list_time_pred_algo, color = 'b', width = width, label = 'Predicted algorithm '+model_names[0])

	if len(pred_algo_2) != 0:
		plt.bar(x + width,  list_time_pred_algo_2, color = 'b', alpha=0.3, width = width, label = 'Predicted algorithm '+model_names[1])
		plt.bar(x + 2*width,  list_time_best_algo, color = 'g', width = width, label = 'Best algorithm')
		plt.xticks(x + width,labels, fontsize=tick_font_size)
	else:
		plt.bar(x + width,  list_time_best_algo, color = 'g', width = width, label = 'Best algorithm')
		plt.xticks(x + width/2,labels, fontsize=tick_font_size)
	
	plt.xlabel('Size of b divided by 10^4', fontsize=tick_font_size+2)
	plt.ylabel('Average computation time in 100ns', fontsize=tick_font_size+2)

	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)
	plt.savefig(fig_path+"bar_times_per_size.png")

	x = np.arange(len(list_time_best_algo))*0.9

	#Plotting the proportion of time each algorithm is predicted
	plt.clf()
	plt.subplots(figsize =(22, 8))
	

	width = 0.2
	
	plt.bar(x, pred_algo_times[0], color = 'b', width = width, label = 'Pred General '+model_names[0])
	plt.bar(x,  pred_algo_times[1], color = 'r', bottom = pred_algo_times[0], width = width, label = 'Pred 2phase '+model_names[0])
	plt.bar(x,  pred_algo_times[2], color = 'g', bottom = np.array(pred_algo_times[0]) + np.array(pred_algo_times[1]), width = width, label = 'Pred 1phase '+model_names[0])

	if len(pred_algo_2) != 0:
		plt.bar(x + width, pred_algo_2_times[0], color = 'b', width = width, alpha=0.66, label = 'Pred General '+model_names[1])
		plt.bar(x + width,  pred_algo_2_times[1], color = 'r', bottom = pred_algo_2_times[0], width = width, alpha=0.66, label = 'Pred 2phase '+model_names[1])
		plt.bar(x + width,  pred_algo_2_times[2], color = 'g', bottom = np.array(pred_algo_2_times[0]) + np.array(pred_algo_2_times[1]), alpha=0.66, width = width, label = 'Pred 1phase '+model_names[1])

		plt.bar(x + 2*width, best_algo_times[0], color = 'b', width = width, alpha=0.3, label = 'GT General ')
		plt.bar(x + 2*width,  best_algo_times[1], color = 'r', bottom = best_algo_times[0], width = width, alpha=0.3, label = 'GT 2phase ')
		plt.bar(x + 2*width,  best_algo_times[2], color = 'g', bottom = np.array(best_algo_times[0]) + np.array(best_algo_times[1]), width = width, alpha=0.3, label = 'GT 1phase')
		plt.xticks(x + width,labels, fontsize=tick_font_size)
	else:
		plt.bar(x + width, best_algo_times[0], color = 'b', width = width, alpha=0.3, label = 'GT General ')
		plt.bar(x + width,  best_algo_times[1], color = 'r', bottom = best_algo_times[0], width = width, alpha=0.3, label = 'GT 2phase ')
		plt.bar(x + width,  best_algo_times[2], color = 'g', bottom = np.array(best_algo_times[0]) + np.array(best_algo_times[1]), width = width, alpha=0.3, label = 'GT 1phase')
		plt.xticks(x + width/2, labels, fontsize=tick_font_size)
	
	#ax.bar(x + 0.50,  list_time_average_algo, color = 'r', width = 0.25)
	plt.xlabel('Size of b divided by 10^4', fontsize=tick_font_size+2)
	plt.ylabel('Percentage', fontsize=tick_font_size+2)

	
	plt.yticks(fontsize=tick_font_size)

	plt.legend(fontsize=tick_font_size)
	plt.savefig(fig_path+"bar_algorithm_perc_per_size.png")


"""


def plot_per_b_size(pred_list, dt_test, model_names, fig_path = "bar_plot"):

	b_sizes = np.concatenate((dt_test[:, 1], dt_test[:, 5]))
	times =  np.concatenate((dt_test[:, 11:14], dt_test[:, 14:17]))

	nb_bar = 15
	min_b_sizes = np.min(b_sizes)
	max_b_sizes = np.max(b_sizes)

	start_ranges = np.arange(min_b_sizes,max_b_sizes+1, int((max_b_sizes-min_b_sizes)/nb_bar))
	
	labels = []

	list_time_per_algo = []
	list_prop_per_algo = []


	for model_num in range(len(pred_list)):

		list_time_per_algo.append([])
		list_prop_per_algo.append([])

		for method_num in range(3):
			list_prop_per_algo[model_num].append([])

	for j in range(len(start_ranges)-1):

		mask_range = np.logical_and(b_sizes < start_ranges[j+1], b_sizes >=  start_ranges[j])



		for model_num in range(len(pred_list)):

			list_time_algo = list_time_per_algo[model_num]
			list_prop_algo = list_prop_per_algo[model_num]

			pred_algo = pred_list[model_num]
			pred_algo_in_range = pred_algo[mask_range]
			


			if len(pred_algo_in_range) != 0:

				if model_num == 0:
					labels.append(str(int(np.round(start_ranges[j]/10000)))+'-'+str(int(np.round(start_ranges[j+1]/10000))))

				times_in_range = times[mask_range,:]

				list_time_algo.append(np.sum(times_in_range[np.arange(0,times_in_range.shape[0]), pred_algo_in_range])/times_in_range.shape[0])
				
				#making the bar for algo
				values, counts = np.unique(pred_algo_in_range, return_counts=True)
			
				for method_num in range(3):
					if not (method_num in values):
						values = np.append(values, method_num)
						counts = np.append(counts, 0)

				for v in range(len(values)):
					list_prop_algo[values[v]].append(counts[v]*100/np.sum(np.array(counts)))




	x = np.arange(len(list_time_per_algo[0]))*0.9

	#make the first plot with average time per size of b
	plt.clf()

	fig = plt.subplots(figsize =(22, 8))
	width = 0.2
	tick_font_size = 18
	alpha = 1/len(pred_list)
	for model_num in range(len(pred_list)):
		
		plt.bar(x + model_num*width, list_time_per_algo[model_num], color = 'k', alpha = 1-model_num*alpha, width = width, label = 'Predicted algorithm '+model_names[model_num])

	plt.xticks(x + (len(pred_list)-1)*width/2, labels, fontsize=tick_font_size)

	plt.xlabel('Size of b divided by 10^4', fontsize=tick_font_size+2)
	plt.ylabel('Average computation time in 100ns', fontsize=tick_font_size+2)

	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)
	plt.savefig(fig_path+"bar_times_per_size.png")

	#Plotting the proportion of time each algorithm is predicted
	plt.clf()


	plt.subplots(figsize =(22, 8))
	
	for model_num in range(len(pred_list)):
		prop_methods = list_prop_per_algo[model_num]

		plt.bar(x + model_num*width, prop_methods[0], color = 'b', alpha = 1-model_num*alpha, width = width, label = 'Pred General '+model_names[model_num])
		plt.bar(x + model_num*width,  prop_methods[1], color = 'r', alpha = 1-model_num*alpha, bottom = prop_methods[0], width = width, label = 'Pred 2phase '+model_names[model_num])
		plt.bar(x + model_num*width,  prop_methods[2], color = 'g', alpha = 1-model_num*alpha, bottom = np.array(prop_methods[0]) + np.array(prop_methods[1]), width = width, label = 'Pred 1phase '+model_names[model_num])
	
	plt.xticks(x + (len(pred_list)-1)*width/2,labels, fontsize=tick_font_size)

	plt.xlabel('Size of b divided by 10^4', fontsize=tick_font_size+2)
	plt.ylabel('Percentage', fontsize=tick_font_size+2)

	
	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)

	plt.savefig(fig_path+"bar_algorithm_perc_per_size.png")

def plot_per_b_size_folds(pred_list, folds, folds_b_size, time_per_fold, model_names, fig_path = "bar_plot"):


	b_list = []

	for i in range(len(folds)):
		fold = folds_b_size[i]
		b_list.append(fold[0])

	b_sizes = np.concatenate(b_list)

	#nb_bar = 15
	nb_bar = 9
	min_b_sizes = np.min(b_sizes)
	max_b_sizes = np.max(b_sizes)

	start_ranges = np.arange(min_b_sizes,max_b_sizes+1, int((max_b_sizes-min_b_sizes)/nb_bar))
	
	labels = []
	list_time_per_algo = []
	list_prop_per_algo = []


	for model_num in range(len(pred_list)):

		list_time_per_algo.append([])
		list_prop_per_algo.append([])

		for method_num in range(3):
			list_prop_per_algo[model_num].append([])



	

	for model_num in range(len(pred_list)):

		for j in range(len(start_ranges)-1):


			prop_algo_folds = []

			for method_num in range(3):
				prop_algo_folds.append([])

			list_time_algo_fold = []

			for fold_nb in range(len(folds)):

				
				fold = folds[fold_nb]
				fold_b_size = folds_b_size[fold_nb]
				b_fold = np.squeeze(fold_b_size[0])


				mask_range = np.logical_and(b_fold < start_ranges[j+1], b_fold >=  start_ranges[j])

				pred_algo_folds = pred_list[model_num]
				pred_algo_fold = pred_algo_folds[fold_nb]
				pred_algo_in_range = pred_algo_fold[mask_range]

				if len(pred_algo_in_range) != 0:

					found_in_range = True

					times = time_per_fold[fold_nb]

					times_in_range = times[mask_range,:]

					list_time_algo_fold.append(np.sum(times_in_range[np.arange(0,times_in_range.shape[0]), pred_algo_in_range])/times_in_range.shape[0])
					
					#making the bar for algo
					values, counts = np.unique(pred_algo_in_range, return_counts=True)
				
					for method_num in range(3):
						if not (method_num in values):
							values = np.append(values, method_num)
							counts = np.append(counts, 0)

					for v in range(len(values)):
						prop_algo_folds[values[v]].append(counts[v]*100/np.sum(np.array(counts)))

			
			list_time_per_algo[model_num].append(np.mean(np.array(list_time_algo_fold)))

			if model_num == 0:
				labels.append(str(int(np.round(start_ranges[j]/10000)))+'-'+str(int(np.round(start_ranges[j+1]/10000))))

			list_prop_per_algo_method = list_prop_per_algo[model_num]

			for i in range(3):	
				list_prop_per_algo_method[i].append(np.mean(np.array(prop_algo_folds[i])))

	x = np.arange(len(list_time_per_algo[0]))*0.9

	#make the first plot with average time per size of b
	plt.clf()

	fig = plt.subplots(figsize =(22, 8))
	width = 0.2
	tick_font_size = 22
	alpha = 1/len(pred_list)
	for model_num in range(len(pred_list)):
		
		plt.bar(x + model_num*width, list_time_per_algo[model_num], color = 'k', alpha = 1-model_num*alpha, width = width, label = 'Predicted algorithm '+model_names[model_num])

	plt.xticks(x + (len(pred_list)-1)*width/2, labels, fontsize=tick_font_size)

	plt.xlabel('Size of b divided by 10^4', fontsize=tick_font_size+2)
	plt.ylabel('Average computation time in 100ns', fontsize=tick_font_size+2)

	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)
	plt.savefig(fig_path+"bar_times_per_size.png")

	#Plotting the proportion of time each algorithm is predicted
	plt.clf()


	plt.subplots(figsize =(22, 8))
	
	for model_num in range(len(pred_list)):
		prop_methods = list_prop_per_algo[model_num]

		plt.bar(x + model_num*width, prop_methods[0], color = 'b', alpha = 1-model_num*alpha, width = width, label = 'Pred General '+model_names[model_num])
		plt.bar(x + model_num*width,  prop_methods[1], color = 'r', alpha = 1-model_num*alpha, bottom = prop_methods[0], width = width, label = 'Pred 2phase '+model_names[model_num])
		plt.bar(x + model_num*width,  prop_methods[2], color = 'g', alpha = 1-model_num*alpha, bottom = np.array(prop_methods[0]) + np.array(prop_methods[1]), width = width, label = 'Pred 1phase '+model_names[model_num])
	
	plt.xticks(x + (len(pred_list)-1)*width/2, labels, fontsize=tick_font_size)

	plt.xlabel('Size of b divided by 10^4', fontsize=tick_font_size+2)
	plt.ylabel('Percentage', fontsize=tick_font_size+2)

	
	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)

	plt.savefig(fig_path+"bar_algorithm_perc_per_size.png")

def plot_MAPE_xnnz_per_b_size(xnnz_pred, dt_test, fig_path = "bar_plot"):

	# --------------------------------------------------------------------
	# This function saves a bar plot in "fig_path" of the MAPE of the
	# the prediction of number of non-zeros elements in the solution
	# over different ranges of size systems
	# --------------------------------------------------------------------


	b_sizes = np.concatenate((dt_test[:, 1], dt_test[:, 5]))
	nb_bar = 15

	min_b_sizes = np.min(b_sizes)
	max_b_sizes = np.max(b_sizes)

	start_ranges = np.arange(min_b_sizes, max_b_sizes+1, int((max_b_sizes-min_b_sizes)/nb_bar))
	

	xnnz_true = np.concatenate((dt_test[:, 4], dt_test[:, 8]))
	labels = []
	MAPEs = []

	for j in range(len(start_ranges)-1):

		mask_range = np.logical_and(b_sizes < start_ranges[j+1], b_sizes >=  start_ranges[j])

		xnnz_pred_in_range = xnnz_pred[mask_range]

		if len(xnnz_pred_in_range) != 0:

			labels.append(str(int(np.round(start_ranges[j]/10000)))+'-'+str(int(np.round(start_ranges[j+1]/10000))))

			xnnz_in_range = xnnz_true[mask_range]

			MAPEs.append(mean_absolute_percentage_error(xnnz_in_range, xnnz_pred_in_range))


	x = np.arange(len(MAPEs))*0.9

	#make the first plot with average time per size of b
	plt.clf()

	fig = plt.subplots(figsize =(22, 8))
	width = 0.2
	tick_font_size = 18
		
	plt.bar(x, MAPEs, color = 'k', width = width)

	plt.xticks(x, labels, fontsize = tick_font_size)

	plt.xlabel('Size of b divided by 10^4', fontsize=tick_font_size+2)
	plt.ylabel('MAPE', fontsize=tick_font_size+2)

	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)
	plt.savefig(fig_path+"bar_MAPES_per_size.png")
	
def plot_MAPE_xnnz_per_xnnz(xnnz_pred, dt_test, fig_path = "bar_plot"):

	# --------------------------------------------------------------------
	# This function saves a bar plot in "fig_path" of the MAPE of the
	# the prediction of number of non-zeros elements in the solution
	# over different ranges of size systems
	# --------------------------------------------------------------------


	b_sizes = np.concatenate((dt_test[:, 4], dt_test[:, 8]))
	nb_bar = 15

	min_b_sizes = np.min(b_sizes)
	max_b_sizes = np.max(b_sizes)

	start_ranges = np.arange(min_b_sizes, max_b_sizes+1, int((max_b_sizes-min_b_sizes)/nb_bar))
	

	xnnz_true = np.concatenate((dt_test[:, 4], dt_test[:, 8]))
	labels = []
	MAPEs = []

	for j in range(len(start_ranges)-1):

		mask_range = np.logical_and(b_sizes < start_ranges[j+1], b_sizes >=  start_ranges[j])

		xnnz_pred_in_range = xnnz_pred[mask_range]

		if len(xnnz_pred_in_range) != 0:

			labels.append(str(int(np.round(start_ranges[j]/10000)))+'-'+str(int(np.round(start_ranges[j+1]/10000))))

			xnnz_in_range = xnnz_true[mask_range]

			MAPEs.append(mean_absolute_percentage_error(xnnz_in_range, xnnz_pred_in_range))


	x = np.arange(len(MAPEs))*0.9

	#make the first plot with average time per size of b
	plt.clf()

	fig = plt.subplots(figsize =(22, 8))
	width = 0.2
	tick_font_size = 18
		
	plt.bar(x, MAPEs, color = 'k', width = width)

	plt.xticks(x, labels, fontsize = tick_font_size)

	plt.xlabel('xnnz', fontsize=tick_font_size+2)
	plt.ylabel('MAPE', fontsize=tick_font_size+2)

	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)
	plt.savefig(fig_path+"bar_MAPES_per_xnnz.png")
	

"""
This function plots the regressions of the time of each algorithm
"""

def plotRegressions(model, xGeneral, x2Phase, x1Phase, trueGeneral, true2Phase, true1Phase, figPath, tick_font_size = 15, fct_to_apply = None):


	if fct_to_apply is None:
		predGeneral = model.predictGeneral(xGeneral)
		pred2Phase = model.predict2Phase(x2Phase)
		pred1Phase = model.predict1Phase(x1Phase)
	else:
		predGeneral = model.predictGeneral(fct_to_apply[0](xGeneral))
		pred2Phase = model.predict2Phase(fct_to_apply[1](x2Phase))
		pred1Phase = model.predict1Phase(fct_to_apply[2](x1Phase))

	methodNames = model.methodNames
	regNames = model.regNames
	colorsMethods = model.colorsMethods
	featureNames = model.featureNames
	

	#plt General
	plt.clf()

	plt.scatter(xGeneral, trueGeneral, c = colorsMethods[0], label = 'Ground truth')
	plt.scatter(xGeneral, predGeneral, c = 'black', label = 'Prediction')
	
	tick_difference = 3

	plt.xticks(fontsize=tick_font_size)
	plt.yticks(fontsize=tick_font_size)
	
	ax = plt.gca()
	ax.xaxis.offsetText.set_fontsize(tick_font_size-tick_difference)
	ax.yaxis.offsetText.set_fontsize(tick_font_size-tick_difference)
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useOffset = True)
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset = True)
	plt.xlabel(featureNames[0], fontsize=tick_font_size+2)
	plt.ylabel("Time in 100ns", fontsize=tick_font_size+2)
	
	plt.legend(fontsize=tick_font_size-3)
	plt.savefig(figPath+"/"+model.name+"_"+methodNames[0]+"_"+regNames[0]+".png", dpi= 300, bbox_inches="tight")


	#plt 2phase 
	plt.clf()
	
	plt.scatter(x2Phase,  true2Phase, c = colorsMethods[1], label = 'Ground truth')
	plt.scatter(x2Phase,  pred2Phase, c = 'black', label = 'Prediction')
	
	plt.xticks(fontsize=tick_font_size)
	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)
	ax = plt.gca()
	ax.xaxis.offsetText.set_fontsize(tick_font_size-tick_difference)
	ax.yaxis.offsetText.set_fontsize(tick_font_size-tick_difference)
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.xlabel(featureNames[1], fontsize=tick_font_size+2)
	plt.ylabel("Time in 100ns", fontsize=tick_font_size+2)
	
	plt.legend(fontsize=tick_font_size-3)
	plt.savefig(figPath+"/"+model.name+"_"+methodNames[1]+"_"+regNames[1]+".png", dpi= 300, bbox_inches="tight")


	#plt 1Phase
	plt.clf()
	
	plt.scatter(x1Phase,  true1Phase, c = colorsMethods[2], label = 'Ground truth')
	plt.scatter(x1Phase,  pred1Phase, c = 'black', label = 'Prediction')
	
	plt.xticks(fontsize=tick_font_size)
	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)
	ax = plt.gca()
	ax.xaxis.offsetText.set_fontsize(tick_font_size-tick_difference)
	ax.yaxis.offsetText.set_fontsize(tick_font_size-tick_difference)
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.xlabel(featureNames[2], fontsize=tick_font_size+2)
	plt.ylabel("Time in 100ns", fontsize=tick_font_size+2)
	
	plt.legend(fontsize=tick_font_size-3)
	plt.savefig(figPath+"/"+model.name+"_"+methodNames[2]+"_"+regNames[2]+".png", dpi= 300, bbox_inches="tight")

"""
This function show in a plot the regression in a logarithm scale in order to see the intersection 
	of the two lines.
"""

def plotIntersections(model, xGeneral, x2Phase, x1Phase, figPath, tick_font_size = 20, fct_to_apply = None):

	methodNames = model.methodNames
	regNames = model.regNames
	colorsMethods = model.colorsMethods
	featureNames = model.featureNames
	"""
	if not os.path.isdir(figPath+"/"+model.name):
		os.mkdir(figPath+"/"+model.name)
	"""
	plt.clf()
	fig, ax = plt.subplots()



	#make the prediction if the feature of the algorithm is specified
	if not (xGeneral is None)  :

		if fct_to_apply is None:
			predGeneral = model.predictGeneral(xGeneral)
		else:
			predGeneral = model.predictGeneral(fct_to_apply[0](xGeneral))
			
		ax.scatter(xGeneral, predGeneral, c = colorsMethods[0], label = methodNames[0])

		


	if  not (x2Phase is None) :
		if fct_to_apply is None:
			pred2Phase = model.predict2Phase(x2Phase)	
		else:
			pred2Phase = model.predict2Phase(fct_to_apply[1](x2Phase))

		ax.scatter(x2Phase, pred2Phase, c = colorsMethods[1], label = methodNames[1])
			


	if  not (x1Phase is None) :

		if fct_to_apply is None:
			pred1Phase = model.predict1Phase(x1Phase)	
		else:
			pred1Phase = model.predict1Phase(fct_to_apply[2](x1Phase))


		ax.scatter(x1Phase, pred1Phase, c = colorsMethods[2], label = methodNames[2])
	

	plt.xlabel(featureNames[2], fontsize=tick_font_size)
		

	
	plt.xticks(fontsize=tick_font_size)
	plt.yticks(fontsize=tick_font_size)
	plt.legend(fontsize=tick_font_size)
	ax.xaxis.set_major_locator(MultipleLocator(1))
	ax.xaxis.set_minor_locator(MultipleLocator(0.1))

	ax.set_yscale('log')
	ax.set_xscale('log')

	plt.ylabel("Time in 100ns",fontsize=tick_font_size)

	plt.savefig(figPath+"/"+model.name+"_intersection.png", dpi= 300, bbox_inches="tight")