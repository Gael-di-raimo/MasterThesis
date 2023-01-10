import os
import numpy as np
import time

from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
from scipy.sparse import eye, identity
from scipy.sparse.linalg import spsolve_triangular, inv
from CMatrix import CMatrix

def compare_cond_matrix(cmatrix_path, matrix, printRes = False):

	if not os.path.isfile(cmatrix_path):
		print("File of the solution was "+cmatrix_path+" not created.")
		exit(1)
	

	cMatrix = CMatrix(cmatrix_path)

	c_scipy_matrix = csc_matrix((cMatrix.data, cMatrix.indices, cMatrix.indptr), shape=(cMatrix.nRow, cMatrix.nCol))
	
	if(c_scipy_matrix.shape[0] != matrix.shape[0]):
		print("Different solution shape! solShape =  "+str(c_scipy_matrix.shape) + " scipyShape = "+str(matrix.shape))

	if  not np.allclose(c_scipy_matrix.todense(), matrix.todense(), atol = 0.001,rtol =0.00001):
		print("Scipy Coo\n" +str(coo_matrix(matrix)))
		print("Mymtx Coo\n"+str(coo_matrix(c_scipy_matrix)))
		mask = matrix.todense()-c_scipy_matrix.todense() != 0
		print(matrix.todense()[mask])
		print(c_scipy_matrix.todense()[mask])
		return False

	if printRes:
		print("Scipy result is the same as the solution.")
		print("Scipy Coo\n" +str(coo_matrix(matrix)))
		print("Mymtx Coo\n"+str(coo_matrix(c_scipy_matrix)))
	return True

def solve(solve_type, isLowerTriangular, mtx_path, factorization_id, solve_id, output_path, testing):

	command = "solver " + str(solve_type)+" "+ str(isLowerTriangular)+ " \""+mtx_path+"\" "+str(factorization_id)+" "+str(solve_id)
	
	command += " \""+output_path+"\" "+str(testing)
	os.system(command)


	if testing:
		L = mmread(mtx_path+"/L"+str(factorization_id).zfill(8)+'.mtx')
		b = mmread(mtx_path+"/b"+str(solve_id).zfill(8)+'.mtx')

		sol = spsolve_triangular(L, b.todense(), lower = True)

		
		
		compare_cond_matrix("sol1phase.txt", csc_matrix(sol), printRes = True)
		if not compare_cond_matrix("solGeneral.txt", csc_matrix(sol)):
			print("Different result for general solve at file rhs = "+str(solve_id)+" with L_id" +str(factorization_id) )
			exit(1)

		if not compare_cond_matrix("sol2phases.txt", csc_matrix(sol)):
			print("Different result for general solve at file rhs_id = "+str(solve_id)+" with L_id" +str(factorization_id) )
			exit(1)

		if not compare_cond_matrix("sol1phase.txt", csc_matrix(sol)):
			print("Different result for general solve at file rhs_id = "+str(solve_id)+" with L_id" +str(factorization_id))
			exit(1)




if __name__ == "__main__":

	L_mult_b_path = 'data/L_multiple_b'
	L_mult_x_path = 'data/L_multiple_x'
	x_mult_L_path = 'data/x_multiple_L'
	mult_L_mult_x_path = 'data/mutliple_L_multiple_x'
	mult_L_mult_b_path = 'data/mutliple_L_multiple_b'

	solve_1 = False
	solve_1_2 = False
	solve_2 = False
	solve_3 = False
	solve_4 = True
	test = False

	#compile the solver
	if os.path.isfile('solver.exe'):
		os.system("del solver.exe")


	os.system("gcc -Wall ../mainSimpleDt.c ../mtx.c ../algo/general.c ../algo/twoPhases.c ../algo/onePhase.c ../utility/heap.c ../features/features.c -o solver")


	if test:
		solve(1, 1, 'data/test/', 10, 10, 'data/test/', 0)
		L = mmread('data/test'+"/L"+str(10).zfill(8)+'.mtx')
		b = mmread('data/test'+"/b"+str(10).zfill(8)+'.mtx')

		sol = spsolve_triangular(L, b.todense(), lower = True)

		print(L.todense())
		print(sol)
		exit(0)


	#read the first dataset

	if solve_1:
		L_id = "0"

		#loop over all possible b
		files = os.listdir(L_mult_b_path)

		for file in files:
			print(file)
			if file[0] == 'b':
				b_id = int(file[1:9])

				solve(1, 1, L_mult_b_path, L_id, b_id, L_mult_b_path, 0)
	if solve_1_2:
		L_id = "0"

		#loop over all possible b
		files = os.listdir(L_mult_x_path)

		for file in files:
				
			if file[0] == 'b':
				b_id = int(file[1:9])
				print("b file id is %d"%(b_id))
				solve(1, 1, L_mult_x_path, L_id, b_id, L_mult_x_path, 0)


	#read the second dataset
	if solve_2:
		#loop over all possible L and b
		files = os.listdir(x_mult_L_path)

		for file in files:

			if file[0] == 'b':

				b_id = int(file[1:9])
				L_id = int(file[1:9])

				solve(1, 1, x_mult_L_path, L_id, b_id, x_mult_L_path, 0)



	#read the third dataset
	if solve_3:
		#loop over all possible L and b

		files = os.listdir(mult_L_mult_x_path)

		for file in files:

			if file[0] == 'b':

				b_id = int(file[1:9])
				L_id = int(file[1:9])

				solve(1, 1, mult_L_mult_x_path, L_id, b_id, mult_L_mult_x_path, 0)

	if solve_4:
		#loop over all possible L and b

		files = os.listdir(mult_L_mult_b_path)

		for file in files:

			if file[0] == 'b':

				3b_id = int(file[1:9])
				L_id = int(file[1:5])*10000
				
				solve(1, 1, mult_L_mult_b_path, L_id, b_id, mult_L_mult_b_path, 0)

