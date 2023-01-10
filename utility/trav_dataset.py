import numpy as np
import os
import time
import multiprocessing as mp
import matplotlib.pylab as plt
import signal

from scipy.io import mmread
from scipy.sparse import csc_matrix
from scipy.sparse import eye





"""
	See the to_save_path to retun a set of different lu_log.tsv files already computed.
		
		returns:
			- a set with the name of datasets already saved
"""

def findAlreadyComputedData(to_save_path):
	to_save_path

	folders = os.listdir(to_save_path)
	folder_set = set()
	folder_set.update(folders)

	return folder_set

"""
	Travel all lulog.tsv file to know the number of different lu_log.tsv files.
		returns:
			- the number of lu_log.tsv files in dataset_path (and its subfolders)
"""

def getNbTotalFilesToRead(dataset_path, log_file_name = "lu_log.tsv" ):

	nbFilesToRead = 0
	toExplore = set()
	folders = os.listdir(dataset_path)

	for file in folders:
		path = dataset_path + "/" + file
		toExplore.add(path)

	while len(toExplore) != 0:

		toExplorePath = toExplore.pop()
		if os.path.isdir(toExplorePath) and os.path.isfile(toExplorePath+"/"+log_file_name):
			nbFilesToRead +=1

		else:
			if os.path.isdir(toExplorePath):
				folders = os.listdir(toExplorePath)

				for file in folders:
					toExplore.add(toExplorePath + "/" + file)

	return nbFilesToRead


def trav_lulog(dataset_path, fct, args = None, nb_folder_to_do = None, to_save_path = None, use_pool = False, remove_ctrl = True, log_file_name = "lu_log.tsv" ):
	
	#some value to see the progress
	nbFilesToRead = getNbTotalFilesToRead(dataset_path)

	#find already computred data in to_save_path
	if to_save_path != None:
		already_computed_data = findAlreadyComputedData(to_save_path)
	else:
		already_computed_data = set()

	nbFilesAlreadyRead = len(already_computed_data)
		
	#get the folders to read from dataset path
	toExplore = set()
	folders = os.listdir(dataset_path)

	#add all folder from the dataset directory
	for file in folders:
		path = dataset_path + "/" + file
		
		if os.path.isdir(path):
			toExplore.add(path)



	#jobs given with the path attributed to a process
	set_lu_log_paths = set()
	jobs_given = 0

	if use_pool:
		#parallel init
		name_jobs = []
		print("Computation will run in "+str(mp.cpu_count())+" core.")
		pool = mp.Pool(mp.cpu_count())


	#BFS of the folders
	nb_folder_done = 0

	print("Start computation at "+str(nbFilesAlreadyRead)+"/"+str(nbFilesToRead))
	start = time.perf_counter()
	while len(toExplore) != 0:

		toExplorePath = toExplore.pop()
		print(toExplorePath)

		if os.path.isdir(toExplorePath) and os.path.isfile(toExplorePath+"/"+log_file_name):

			lu_log_path = toExplorePath
			splittedPath = toExplorePath.split("/")
			data_name = splittedPath[-1]

			if use_pool == True:

				#do the jobs in parallel

				if jobs_given < mp.cpu_count() and len(toExplore) != 0:
					#give a path to a process

					if not (data_name in already_computed_data):

						set_lu_log_paths.add(lu_log_path)
						already_computed_data.add(data_name)
						jobs_given +=1

						name_jobs.append(data_name)

						if to_save_path != None and not os.path.exists( to_save_path+"/"+data_name):
							#Create the path for the product matrices of the dataset if it doesn't exist
							os.makedirs(to_save_path+"/"+data_name)
				
				if jobs_given ==  mp.cpu_count():

					jobs_given = 0

					list_path = list(set_lu_log_paths)
					set_lu_log_paths = set()

					print("Start reading "+str(name_jobs))

					results = pool.map(fct, list_path)
					name_jobs = []
					nbFilesAlreadyRead += len(list_path)
					nb_folder_done += len(list_path)

			else:
				
				#do not in parrallel
				if to_save_path != None and not os.path.exists( to_save_path+"/"+data_name):
					#Create the path for the product matrices of the dataset if it doesn't exist
					os.makedirs(to_save_path+"/"+data_name)

				if not (data_name in already_computed_data):
					t2= time.perf_counter()
					

					print("Start reading "+data_name)
					if args != None:
						fct(lu_log_path, args)				
					else:
						fct(lu_log_path)

					already_computed_data.add(data_name)
					nbFilesAlreadyRead +=1
					nb_folder_done += 1

			
			print("Files read = "+str(nbFilesAlreadyRead)+"/"+str(nbFilesToRead))

			if len(toExplore) ==  0 and use_pool == True:

				#check if jobs to do
				if jobs_given != 0:

					print("Start reading "+str(name_jobs))
					name_jobs = []

					list_path = list(set_lu_log_paths)
					nbFilesAlreadyRead += len(list_path)
					results =  pool.map(fct, list_path)
					
					nb_folder_done += len(list_path)

					set_lu_log_paths = set()

			
			if nb_folder_to_do != None and nb_folder_done >= nb_folder_to_do:

				if use_pool:
					pool.close()

				exit(0)

		else:

			#if not a file add all sub folders
			folders = os.listdir(toExplorePath)

			for folder in folders:

				if os.path.isdir(toExplorePath + "/" + folder):
					toExplore.add(toExplorePath + "/" + folder)

	if use_pool:
		pool.close()

	time_taken = time.perf_counter() - start
	print("Time taken is: "+str(time_taken))

	return 0

def trav_outfile(dataset_path, fct, args = None, nb_folder_to_do = None, use_pool = False, remove_ctrl = True):


	already_computed_data = set()
	#get the folders to read from dataset path
	toExplore = set()
	folders = os.listdir(dataset_path)

	#add all folder from the dataset directory
	for file in folders:
		path = dataset_path + "/" + file
		toExplore.add(path)

	#jobs given with the path attributed to a process
	set_out_paths = set()
	jobs_given = 0

	if use_pool:
		#parallel init
		name_jobs = []
		print("Computation will run in "+str(mp.cpu_count())+" core.")
		pool = mp.Pool(mp.cpu_count())


	#BFS of the folders
	nb_file_done = 0
	print("Start computation")

	start = time.perf_counter()

	while len(toExplore) != 0:

		toExplorePath = toExplore.pop()

		#check is out file

		if toExplorePath[len(toExplorePath)-3:len(toExplorePath)] == "out":
			print(toExplorePath)
			splittedPath = toExplorePath.split("/")
			data_name = splittedPath[-1]
			data_name = data_name[0:-4]

			if use_pool == True:

				#do the jobs in parallel

				if jobs_given < mp.cpu_count() and len(toExplore) != 0:
					#give a path to a process

					if not (data_name in already_computed_data):

						set_out_paths.add(toExplorePath)
						already_computed_data.add(data_name)
						jobs_given +=1

						name_jobs.append(data_name)
				
				if jobs_given ==  mp.cpu_count():

					jobs_given = 0

					list_path = list(set_out_paths)
					set_out_paths = set()

					print("Start reading "+str(name_jobs))

					results = pool.map(fct, list_path)
					name_jobs = []
					nb_file_done += len(list_path)

			else:

				#do not in parrallel
				if not (data_name in already_computed_data):

					print("Start reading "+data_name)
					if args != None:

						fct(toExplorePath, args)
					
					else:
						fct(toExplorePath)

					already_computed_data.add(data_name)
					nb_file_done += 1


			if len(toExplore) ==  0 and use_pool == True:

				#check if jobs to do
				if jobs_given != 0:

					print("Start reading "+str(name_jobs))
					name_jobs = []

					list_path = list(set_out_paths)
					results =  pool.map(fct, list_path)
					
					nb_file_done += len(list_path)

					set_out_paths = set()

			print("Files read = "+str(nb_file_done))
			
			if nb_folder_to_do != None and nb_folder_done >= nb_folder_to_do:

				if use_pool:
					pool.close()

				exit(0)

		else:
			if os.path.isdir(toExplorePath):
				folders = os.listdir(toExplorePath)

				for file in folders:

					if os.path.isfile(toExplorePath + "/" + file):
						toExplore.add(toExplorePath + "/" + file)

	if use_pool:
		pool.close()

	time_taken = time.perf_counter() - start
	print("Time taken is: "+str(time_taken))

	return 0

"""
	-------- Testing part of the file --------

"""
def test_trav(lu_log_path):
	print("test_trav with:"+lu_log_path)

def test_features():

	path ="E:/Cour Master INFO/TFE/code/codeVerifyFeatures/dataset"
	trav_lulog(path, test_trav)
	return
	
	

if __name__ == "__main__":
	test_features()
