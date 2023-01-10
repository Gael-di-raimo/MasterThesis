import os
import numpy as np
import time

from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve_triangular, inv

from test_python.CMatrix import CMatrix

def print_array_21_decimals(array):

	print("[", end ='')
	for i in range(len(array)-1):
		print("%.21lf, " % array[i], end ='')

	print("%.21lf]" % array[len(array)-1],end ='')

def createNewRhs(matrix, rhs):

	new_rhs = np.zeros((matrix.shape[0],1)) 
	
	if matrix.shape[0] > rhs.shape[0]:
		new_rhs[0:rhs.shape[0]] = rhs
	else:
		new_rhs = rhs[0:matrix.shape[0]]

	new_rhs_path = "tmp_rhs.mtx"

	mmwrite(new_rhs_path, coo_matrix(new_rhs))

	return new_rhs, new_rhs_path

def delNewRhs():
	os.remove("tmp_rhs.mtx")

def compare_cond_matrix(cmatrix_path, matrix, printRes = False):

	if not os.path.isfile(cmatrix_path):
		print("File of the solution was "+cmatrix_path+" not created.")
		exit(1)
	

	cMatrix = CMatrix(cmatrix_path)

	c_scipy_matrix = csc_matrix((cMatrix.data, cMatrix.indices, cMatrix.indptr), shape=(cMatrix.nRow, cMatrix.nCol))
	
	if(c_scipy_matrix.shape[0] != matrix.shape[0]):
		print("Different solution shape! solShape =  "+str(c_scipy_matrix.shape) + " scipyShape = "+str(matrix.shape))

	if  not np.allclose(c_scipy_matrix.todense(), matrix.todense()):
		print("Scipy Coo\n" +str(coo_matrix(matrix)))
		print("Mymtx Coo\n"+str(coo_matrix(c_scipy_matrix)))
		return False

	if printRes:
		print("Scipy result is the same as the solution.")
		print("Scipy Coo\n" +str(coo_matrix(matrix)))
		print("Mymtx Coo\n"+str(coo_matrix(c_scipy_matrix)))
	return True

def solve(matrix, rhs , mtx_path, factorization_id, solve_id, output_path, triangularForm, solve_type, printSolvePath = False, testing = False):

	if printSolvePath:
		print("Testing : factorization_id = "+str(factorization_id)+", solve_id = " +str(solve_id) +", and solveType = "+ str(solve_type)+" at path "+output_path)

	#see if there is a problem of shape due the other matrices involved
	if matrix != None and  rhs != None and rhs.shape[0] != matrix.shape[0]:
		rhs, rhs_path = createNewRhs(matrix, rhs)


	if testing:
		testing = 1
	else:
		testing = 0

	command = "test " + str(solve_type)+ " \""+mtx_path+"\" "+str(factorization_id)+" "+str(solve_id)
	
	command += " \""+output_path+"\" "+str(testing)

	time_1 = time.time()
	os.system(command)
	time_2 = time.time()
	
	if testing:
		filled_f_id = str(factorization_id).zfill(8)
		
		U_path = mtx_path+"/U"+filled_f_id+".mtx"
		L_path = mtx_path+"/L"+filled_f_id+".mtx"

		#compare inverted and other solve
		comp_inv_solve = False
		#compare with scipy results
		if solve_type:

			time_3 = time.time()
			sol = spsolve_triangular(matrix, rhs, lower = solve_type)
			time_4 = time.time()

			sol_adjusted = sol

			matrix2 = mmread(U_path).tocsr()

			if sol.shape[0] != matrix2.shape[0]:
				print("Different solution shape and mtx2 ! solShape =  "+str(sol.shape) + " matrix2Shape = "+str(matrix2.shape))
				sol_adjusted, sol_path = createNewRhs(matrix2, sol)

			time_5 = time.time()
			sol_2 =  spsolve_triangular(matrix2, sol_adjusted, lower = not solve_type)
			time_6 = time.time()
		else:
			time_3 = time.time()
			sol = spsolve_triangular(matrix.transpose(), rhs, lower = not solve_type)
			time_4 = time.time()

			if comp_inv_solve:
				sol_test = rhs.transpose()*(inv(matrix))
				sol_test = sol_test.transpose()

				if not np.allclose(sol, sol_test):
					print("Error different value, scipy inverted solution =\n" + str(coo_matrix(sol)) + "\n, scipy triangular solve = \n" +str(coo_matrix(sol_test))) 
					exit(1)

			
			matrix2 = mmread(L_path).tocsr()

			sol_adjusted = sol

			if sol.shape[0] != matrix2.shape[0]:
				print("Different solution shape and mtx2 ! solShape =  "+str(sol.shape) + " matrix2Shape = "+str(matrix2.shape)+ " matrix1Shape = "+str(matrix.shape))
				sol_adjusted, sol_path = createNewRhs(matrix2, sol)

			time_5 = time.time()
			sol_2 = spsolve_triangular(matrix2.transpose(), sol_adjusted, lower = solve_type)
			time_6 = time.time()
			if comp_inv_solve:

				sol_test2 = sol_adjusted.transpose()*(inv(matrix2))
				sol_test2 = sol_test2.transpose()

				if not np.allclose(sol_2, sol_test2):
					print("Error different value, scipy inverted solution =\n" + str(coo_matrix(sol_2)) + "\n, scipy triangular solve = \n" +str(coo_matrix(sol_test2))) 
					exit(1)


		#compare general solve results
		if not compare_cond_matrix("solGeneral.txt", csc_matrix(sol)):
			print("Different result for general solve at file rhs = "+rhs_path+" with " +m_path + ", solve type "+str(solve_type)+ ", triangularForm "+str(triangularForm))
			exit(1)
		else:
			print("General Solution1 ok")


		if not compare_cond_matrix("solGeneral2.txt", csc_matrix(sol_2)):
			print("Different result for general solve (second part) at file rhs = "+rhs_path+" with " +m_path + ", solve type "+str(solve_type)+ ", triangularForm "+str(triangularForm))
			exit(1)
		else:
			print("General Solution2 ok")



		#compare twoPhases results
		if not compare_cond_matrix("sol2phases.txt", csc_matrix(sol)):
			print("Different result for twoPhases solve at file rhs = "+rhs_path+" with " +m_path + ", solve type "+str(solve_type)+ ", triangularForm "+str(triangularForm))
			exit(1)
		else:
			print("2Phases Solution 1 ok")

		if not compare_cond_matrix("sol2phases2.txt", csc_matrix(sol_2)):
			print("Different result for twoPhases solve (second part) at file rhs = "+rhs_path+" with " +m_path + ", solve type "+str(solve_type)+ ", triangularForm "+str(triangularForm))
			exit(1)
		else:
			print("2Phases Solution 2 ok")

		if not compare_cond_matrix("sol1phase.txt", csc_matrix(sol)):
			print("Different result for onePhase solve at file rhs = "+rhs_path+" with " +m_path + ", solve type "+str(solve_type)+ ", triangularForm "+str(triangularForm))
			exit(1)
		else:
			print("1Phase Solution 1 ok")

		if not compare_cond_matrix("sol1phase2.txt", csc_matrix(sol_2)):
			print("Different result for onePhase solve (second part) at file rhs = "+rhs_path+" with " +m_path2 + ", solve type "+str(solve_type)+ ", triangularForm "+str(triangularForm))
			exit(1)
		else:
			print("1Phase Solution 2 ok")

		
		os.remove("solGeneral.txt")
		os.remove("solGeneral2.txt")
		os.remove("sol2phases.txt")
		os.remove("sol2phases2.txt")
		os.remove("sol1phase.txt")
		os.remove("sol1phase2.txt")
		print("MySolveTime = "+str(time_2-time_1))
		print("ScypyTime = "+str(time_4-time_3 + time_6 -time_5))
	
	if matrix != None and rhs != None and rhs.shape[0] != matrix.shape[0]:
		delNewRhs()
	
def process_tsv(tsv_path, args):

	#create path to read mtx
	print("Opening "+tsv_path+".")

	tsv = open(tsv_path)
	tsv_lines = tsv.readlines()

	max_samples = args[3]

	#retreive path
	split_path = tsv_path.split("/")
	len_path = len(tsv_path) - len(split_path[-1])

	path = tsv_path[0:len_path-1]

	#create path for timers
	output_path = args[0]
	dataset_path = args[1]

	testing = args[2]

	#create a corresponding save path
	tmp_path = path[len(dataset_path):len(path)]
	output_path = output_path + tmp_path


	output_path_split = output_path.split("/")
	append_path = output_path_split[0]

	for i in range(1, len(output_path_split)):
		print(append_path)
		append_path = append_path+"/"+output_path_split[i]

		if not os.path.isdir(append_path):
			os.mkdir(append_path)


	#reading te state when the previous run stopped
	if max_samples == -1:
		highest_fid = -1
		highest_sid = -1

		if os.path.isfile(output_path+"/timers.txt"):
		
			timer_file = open(output_path+"/timers.txt")
			timer_lines = timer_file.readlines()
		
			for line in timer_lines:
				line_split = line.split()

				if int(line_split[1]) > highest_fid:
					highest_fid = int(line_split[1])

				if int(line_split[2]) > highest_sid:
					highest_sid = int(line_split[2])

			timer_file.close()

	else:
		highest_fid = -1
		highest_sid = -1

		if os.path.isfile(output_path+"/index_samples.csv"):

			index_samples = np.loadtxt(output_path+"/index_samples.csv", delimiter = ",").astype(int)

			#file already run
			if os.path.isfile(output_path+"/timers.txt"):
				timer_file = open(output_path+"/timers.txt")
				timer_lines = timer_file.readlines()

				index_samples = index_samples[len(timer_lines)+1: len(index_samples)]

				timer_file.close()
		
		else:
			nb_samples = int((len(tsv_lines)-2)/4)

			if nb_samples > max_samples:
				rng = np.random.default_rng()
				index_samples = rng.choice(range(0,nb_samples), max_samples, replace = False)

				np.savetxt(output_path+'/index_samples.csv', index_samples, delimiter=',', fmt = '%d')

			else:
				index_samples = np.arange(0, nb_samples)
				np.savetxt(output_path+'/index_samples.csv', index_samples, delimiter=',', fmt = '%d')

		index_lines = index_samples*4+1
		
		tsv_lines = np.array(tsv_lines)
		tsv_lines = tsv_lines[index_lines]
		print(len(tsv_lines))


	#reading the matrix and calling the solvers
	for line in tsv_lines:
		time_1 = time.time()
		line_split = line.split()

		if line[0] != "b" and line[0] !="e":

			if int(line_split[1]) == 0 or max_samples != -1:

				#get the ids
				factorization_id = 	int(line_split[10])
				solve_id = 	int(line_split[11])

				# check if not already solved
				if factorization_id >= highest_fid and solve_id > highest_sid:

					solve_type = line_split[0]

					filled_f_id = str(factorization_id).zfill(8)
					filled_s_id = str(solve_id).zfill(8)

					L = None
					U = None
					b = None

					if testing:
						b_path = path+"/b"+filled_s_id+".mtx"
						y_path = path+"/y"+filled_s_id+".mtx"
						L_path = path+"/L"+filled_f_id+".mtx"
						U_path = path+"/U"+filled_f_id+".mtx"

						time_2 = time.time()
						b = mmread(b_path).todense()
						L = mmread(L_path).tocsr()
						U = mmread(U_path).tocsr()
						time_3 = time.time()

						print("Time load "+str(time_3-time_2))
					else:

						time_2 = time.time()
						#solve for type LUx = b
						if int(solve_type) == 1:

							#Solving Ly=b then Ux=y						
							solve(L, b, path, factorization_id, solve_id, output_path, True, 1, printSolvePath = True, testing = testing)
							
						else:
							#Solving for type y^tU = b^t then x^tL = y^t
							solve(U, b, path, factorization_id, solve_id, output_path, False, 0, printSolvePath = True, testing = testing)
						
						time_3 = time.time()
						print("Time computation "+str(time_3-time_2))
						print("Total time "+str(time_3-time_1))
					
				else:
					filled_f_id = str(factorization_id).zfill(8)
					filled_s_id = str(solve_id).zfill(8)
					
					b_path = path+"/b"+filled_s_id+".mtx"
					y_path = path+"/y"+filled_s_id+".mtx"
					L_path = path+"/L"+filled_f_id+".mtx"
					U_path = path+"/U"+filled_f_id+".mtx"

					print("Already done "+tsv_path+": b"+filled_s_id+".mtx with L"+filled_f_id+".mtx, U"+filled_f_id+".mtx")


	tsv.close()

def trav_tsv(dataset_path, fct, args = None):


	toExplore = set()
	folders = os.listdir(dataset_path)

	for file in folders:
		path = dataset_path + "/" + file
		toExplore.add(path)

	while len(toExplore) != 0:

		toExplorePath = toExplore.pop()
		print(toExplorePath)
		if os.path.isfile(toExplorePath+"/lu_log.tsv"):
			
			if args == None:
				fct(toExplorePath+"/lu_log.tsv")
			else:
				fct(toExplorePath+"/lu_log.tsv", args)
		else:
			if os.path.isdir(toExplorePath):
				folders = os.listdir(toExplorePath)

				for file in folders:
					toExplore.add(toExplorePath + "/" + file)


	return 0
	

if __name__ == "__main__":

	travDataset = True
	max_samples = 4000

	if travDataset:

		if(os.path.isfile("test.exe")):
			os.system("del test.exe")

		os.system("gcc -Wall main.c mtx.c algo/general.c algo/twoPhases.c algo/onePhase.c utility/heap.c features/features.c -o test")

		data_path ="dt"

	
		
		output_path = "../generated_times_features"


		testing = False
		args = []
		args.append(output_path)
		args.append(data_path)
		args.append(testing)
		args.append(max_samples)

		trav_tsv(data_path, process_tsv, args = args)

	

	