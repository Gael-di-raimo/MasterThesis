import os
import numpy as np

from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve_triangular, inv
from CMatrix import CMatrix

def print_array_21_decimals(array):

	print("[", end ='')
	for i in range(len(array)-1):
		print("%.21lf, " % array[i], end ='')

	print("%.21lf]" % array[len(array)-1],end ='')

"""
	This function creates a new rhs if the dimension does not corresponds, this means
		it will either add zeros to the rhs or removes some values.
"""
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
	print("Removing tmpfile")
	os.remove("tmp_rhs.mtx")

def compare_cond_matrix(cmatrix_path, matrix, printRes = False):

	if not os.path.isfile(cmatrix_path):
		print("File of the c matrix does not exist: " +str(cmatrix_path))
		exit(1)

	cMatrix = CMatrix(cmatrix_path)

	c_scipy_matrix = csc_matrix((cMatrix.data, cMatrix.indices, cMatrix.indptr), shape=(cMatrix.nRow, cMatrix.nCol))
	if( c_scipy_matrix.todense().shape[0] != matrix.todense().shape[0]):
		print("Error got different shape solution.")
		print("Scipy shape\n" +str(matrix.todense().shape[0]))
		print("Mymtx shape\n"+str(c_scipy_matrix.todense().shape[0]))
		return True

	if  not np.allclose(c_scipy_matrix.todense(), matrix.todense()):
		print("Scipy Coo\n" +str(coo_matrix(matrix)))
		print("Mymtx Coo\n"+str(coo_matrix(c_scipy_matrix)))
		return False

	if printRes:
		print("Scipy result is the same as the solution.")
		print("Scipy Coo\n" +str(coo_matrix(matrix)))
		print("Mymtx Coo\n"+str(coo_matrix(c_scipy_matrix)))

	return True

def solve(matrix, rhs, m_path, rhs_path, times_path, triangularForm, solve_type, printSolvePath = False, testing = False):

	if printSolvePath:
		print("Testing : rhs = "+rhs_path+" with " +m_path +", solve type: "+str(solve_type))

	#see if there is a problem of shape due the other matrices involved
	if rhs.shape[0] != matrix.shape[0]:
		rhs, rhs_path = createNewRhs(matrix, rhs)

	if triangularForm != 1 or int(solve_type) != 1:
		print("Should work testing")
	

	if triangularForm:
		triangularForm = 1
	else:
		triangularForm = 0

	command = "test " + str(triangularForm) + " " + str(solve_type)
	
	command += " \""+m_path+"\" \""+rhs_path+"\" \""+times_path+"\""

	os.system(command)

	if testing:

		#compare with scipy results
		if solve_type:
			sol = spsolve_triangular(matrix, rhs, lower = triangularForm)
		else:

			sol_2 = spsolve_triangular(matrix.transpose(), rhs, lower = not triangularForm)
			sol = rhs.transpose()*(inv(matrix))
			sol = sol.transpose()

			if not np.allclose(sol, sol_2) :
				print("Error different value, scipy inverted solution =\n" + str(coo_matrix(sol)) + "\n, scipy triangular solve = \n" +str(coo_matrix(sol_2))) 
				exit(1)

		if not compare_cond_matrix("sol1phase.txt", csc_matrix(sol)):
			print("Different result for general solve at file rhs = "+rhs_path+" with " +m_path + ", solve type "+str(solve_type)+ ", triangularForm "+str(triangularForm))
			exit(1)

	os.remove("sol1phase.txt")

	if rhs.shape[0] != matrix.shape[0]:
		delNewRhs()

def process_tsv(tsv_path, args):

	#create path to read mtx
	print("Opening "+tsv_path+".")

	tsv = open(tsv_path)
	tsv_lines = tsv.readlines()

	split_path = tsv_path.split("/")
	len_path = len(tsv_path) - len(split_path[-1])

	path = tsv_path[0:len_path-1]

	#create path for timers
	times_path = args[0]
	dataset_path = args[1]

	testing = args[2]

	tmp_path = path[len(dataset_path):len(path)]

	times_path = times_path + tmp_path


	times_path_split = times_path.split("/")
	append_path = times_path_split[0]

	for i in range(1, len(times_path_split)):
		print(append_path)
		append_path = append_path+"/"+times_path_split[i]

		if not os.path.isdir(append_path):
			os.mkdir(append_path)

	
	#reading the matrix and calling the solvers
	for line in tsv_lines:
		line_split = line.split()

		if line[0] != "b" and line[0] !="e":

			if int(line_split[1]) == 0:

				solve_type = line_split[0]

				factorization_id = 	int(line_split[10])
				solve_id = 	int(line_split[11])

				filled_f_id = str(factorization_id).zfill(8)
				filled_s_id = str(solve_id).zfill(8)
				
				b_path = path+"/b"+filled_s_id+".mtx"
				y_path = path+"/y"+filled_s_id+".mtx"
				L_path = path+"/L"+filled_f_id+".mtx"
				U_path = path+"/U"+filled_f_id+".mtx"

				if testing:

					b = mmread(b_path).todense()
					y = mmread(y_path).todense()
					L = mmread(L_path).tocsr()
					U = mmread(U_path).tocsr()

				#print("Testing "+tsv_path+": b"+filled_s_id+".mtx with L"+filled_f_id+".mtx")

				#solve for type LUx = b
				if int(solve_type) == 1:

					#Solving Ly=b
					solve(L, b, L_path, b_path, times_path+"/algorithm_times_L.txt", True, 1, printSolvePath = True, testing = True)

					#Solving Ux=y
					solve(U, y, U_path, y_path, times_path+"/algorithm_times_U.txt", False, 1, printSolvePath = True, testing = True)

					"""
					#Solving Ly=b
					print("Testing "+tsv_path+": b"+filled_s_id+".mtx with L"+filled_f_id+".mtx")

					#see if there is a problem of shape due the other matrices involved
					if b.shape[0] != L.shape[0]:
						b = createNewRhs(L, b)

					os.system("test 1 1 \""+L_path+"\" \""+b_path+"\" \""+times_path+"\"/algorithm_times_L.txt")

					#if the purpose is testing it will compare with scipy results
					if testing:	

						my_y = spsolve_triangular(L, b, lower = True)
						#x = spsolve_triangular(U, y, lower = False)
						if compare_solve_general(my_y):
							print("Different result for general solve at file "+tsv_path+" for matrix "+L_path+ " and "+b_path)
							exit(1)


					print("Testing "+tsv_path+": b"+filled_s_id+".mtx with U"+filled_f_id+".mtx")
					os.system("test 0 1 \""+U_path+"\" \""+y_path+"\" \""+times_path+"\"/algorithm_times_U.txt")
					if b.shape[0] != L.shape[0]:
						#Problem of shape due the other matrices involved
					if testing:
						x = spsolve_triangular(U, y, lower = False)

						if compare_solve_general(x):
							print("Different result for general solve at file "+tsv_path+" for matrix "+U_path+ " and "+y_path)
							exit(1)
					"""
				else:
					#Solving for type y^tU = b^t
					solve(U, b, U_path, b_path, times_path+"/algorithm_times_U.txt", False, 0, printSolvePath = True, testing = True)
					
					#Solving for type x^tL = y^t
					solve(L, y, L_path, y_path, times_path+"/algorithm_times_L.txt", True, 0, printSolvePath = True, testing = True)

					"""
					print("Testing "+tsv_path+": b"+filled_s_id+".mtx with L"+filled_f_id+".mtx")
					os.system("test 0 0 \""+U_path+"\" \""+b_path+"\" \""+times_path+"\"/algorithm_times_U.txt")


					print("Testing "+tsv_path+": b"+filled_s_id+".mtx with U"+filled_f_id+".mtx")
					os.system("test 1 0 \""+L_path+"\" \""+y_path+"\" \""+times_path+"\"/algorithm_times_L.txt")
					
					if testing:
						if b.shape[0] != U.shape[0]:
							new_b = np.zeros((U.shape[0],1)) 
							
							if U.shape[0] > b.shape[0]:
								new_b[0:b.shape[0]] = b
							else:
								new_b = b[0:U.shape[0]]

							b = new_b
							

						y = spsolve_triangular(U.transpose(), b, lower = True)

						y = y.transpose()
						y_2 = b.transpose()*(inv(U))

						print("Error different value, general solution =\n" + str(coo_matrix(y)) + "\n, scipysol = \n" +str(coo_matrix(y_2))) 
						if not np.array_equal(y, y_2) :

							print("Error different value, general solution =\n" + str(coo_matrix(y)) + "\n, scipysol = \n" +str(coo_matrix(y_2))) 
							exit(1)
					"""

	tsv.close()

	print("Test finished withour errors.")

def trav_tsv(dataset_path, fct, args = None):


	toExplore = set()
	folders = os.listdir(dataset_path)

	for file in folders:
		path = dataset_path + "/" + file
		toExplore.add(path)

	while len(toExplore) != 0:

		toExplorePath = toExplore.pop()
		if toExplorePath[len(toExplorePath)-3:len(toExplorePath)] == "tsv":
			
			if args == None:
				fct(toExplorePath)
			else:
				fct(toExplorePath, args)
		else:
			if os.path.isdir(toExplorePath):
				folders = os.listdir(toExplorePath)

				for file in folders:
					toExplore.add(toExplorePath + "/" + file)


	return 0
	

if __name__ == "__main__":

	test_data = True


	if test_data:
		if os.path.isfile("test.exe"):
			os.system("del test.exe")

		os.system("gcc ../main.c ../mtx.c ../algo/general.c ../algo/twoPhases.c ../algo/onePhase.c ../utility/heap.c ../headers/heap.h -o test")
		
		data_path ="E:/Cour Master INFO/TFE/instance_samples"
		timer_path = "E:/Cour Master INFO/TFE/timers"

		testing = True
		args = []
		args.append(timer_path)
		args.append(data_path)
		args.append(testing)

		trav_tsv(data_path, process_tsv, args = args)
	
	else:
		if os.path.isfile("test.exe"):
			os.system("del test.exe")

		os.system("gcc ../main.c ../mtx.c ../algo/general.c ../algo/twoPhases.c ../algo/onePhase.c ../utility/heap.c ../headers/heap.h -o test")
		times_path = "../TmpTest/times/"

		b_path = "../TmpTest/b00000098.mtx"
		L_path = "../TmpTest/L00000097.mtx"

		b_path = "../TmpTest/b00000038.mtx"
		L_path = "../TmpTest/L00000036.mtx"

		"""
		b_path = "../TmpTest/myb.mtx"
		L_path = "../TmpTest/myL.mtx"
		"""

		b = mmread(b_path).todense()
		L = mmread(L_path).tocsr()

		print(b.shape)
		print(L.shape)
		if b.shape[0] != L.shape[0]:
			b, b_path = createNewRhs(L, b)
			
		print(b.shape)
		print("Coo"+str(coo_matrix(b)))

		print("Sending the command")
		os.system("test 1 1 \""+L_path+"\" \""+b_path+"\" \""+times_path+"\"/algorithm_times_L.txt")
		print("Command finished")

		y = spsolve_triangular(L, b, lower = True)
		
		if not compare_cond_matrix("sol1phase.txt", csc_matrix(y), printRes = True):
			print("Different result for general solve at current folder for matrix "+L_path+ " and "+b_path)
			exit(1)
	