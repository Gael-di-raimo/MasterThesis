#include "../headers/algorithm.h"


double* generalAlgorithmNotSortedMatrix(CondMatrix* triangularMatrix, void* rhs, int solveType, int triangularForm, bool rhsIsCondMatrix){
	//what if L is 0 ? You divide by zeros ? => try a simple matrix in python with
 	//this problem to see how it deal with it !
	
	bool debug = false;

	if(debug){
		printf("In the not sorted algorithm.\n");
	}

	//read rhs to 'array'
	double* solution = (double*) malloc(sizeof(double)*(triangularMatrix->nRow));

	if(solution == NULL){
		fprintf(stderr,"Error while allocating the solution in general algorithm.");
		return NULL;
	}

	//initialiazing the solution to the rhs
	//two possibilities either travel two times or put an if in it to put a value of rhs or zeros
	int64_t i;
	
	if(rhsIsCondMatrix){

		CondMatrix* rhsCond = (CondMatrix*) rhs;
		//intialize the solution with 0
		for(i = 0; i < rhsCond->nRow; i++){
			solution[i] = 0;
		}

		//adding the value of rhs to the temporary solution

		for(i = 0; i < rhsCond->nnz; i++){
			solution[rhsCond->indices[i]] = rhsCond->data[i];
		}
	}
	else{
		double* rhsArray = (double*) rhs;

		//intialize the solution to the rhs
		for(i = 0; i < triangularMatrix->nRow; i++){
			solution[i] = rhsArray[i];
		}

	}
	
	if(debug){
		printf("TmpSol:[\n");
		for(i = 0; i < triangularMatrix->nRow; i++){
			printf("\t%lf,\n",solution[i]);
		}
		printf("\t]\n");
	}

	int64_t k = 0, indexDiagElement;
	double valMatrix, solution_i;


	if(solveType == triangularForm){
		for(i = 0; i < triangularMatrix->nRow; i++){
			
			//find the diagonal element to compute the final value of solution[i]
			for(k = triangularMatrix->indPtr[i]; k < triangularMatrix->indPtr[i+1]; k++){
				
				if(triangularMatrix->indices[k] == i){

					//computing the value of solution[i] only if valMatrix != 0 (else leads to infinity) 
					indexDiagElement = k;

					valMatrix = triangularMatrix->data[k];

					if(valMatrix != 0){
						solution[i] = solution[i]/valMatrix;
					}
				}

			}


			solution_i = solution[i];

			//exploring the remaining element in the row/col of the triangular matrix

			//dividing into two loops to avoid putting an if in one loop

			for(k = triangularMatrix->indPtr[i]; k < indexDiagElement; k++){

				solution[triangularMatrix->indices[k]] = solution[triangularMatrix->indices[k]] - solution_i*triangularMatrix->data[k];

			}

			for(k = indexDiagElement + 1; k < triangularMatrix->indPtr[i+1]; k++){

				solution[triangularMatrix->indices[k]] = solution[triangularMatrix->indices[k]] - solution_i*triangularMatrix->data[k];

			}

		}
	}
	else{
	
		for(i = triangularMatrix->nRow - 1; i >= 0; i--){
			//find the diagonal element to compute the final value of solution[i]
			for(k = triangularMatrix->indPtr[i]; k < triangularMatrix->indPtr[i+1]; k++){
				
				if(triangularMatrix->indices[k] == i){

					//computing the value of solution[i] only if valMatrix != 0 (else leads to infinity) 
					indexDiagElement = k;

					valMatrix = triangularMatrix->data[k];

					if(valMatrix != 0){
						solution[i] = solution[i]/valMatrix;
					}
				}

			}


			solution_i = solution[i];

			//exploring the remaining element in the row/col of the triangular matrix

			//dividing into two loops to avoid putting an if in one loop

			for(k = triangularMatrix->indPtr[i]; k < indexDiagElement; k++){

				solution[triangularMatrix->indices[k]] = solution[triangularMatrix->indices[k]] - solution_i*triangularMatrix->data[k];

			}

			for(k = indexDiagElement + 1; k < triangularMatrix->indPtr[i+1]; k++){

				solution[triangularMatrix->indices[k]] = solution[triangularMatrix->indices[k]] - solution_i*triangularMatrix->data[k];

			}

		}
	}

	return solution;
}

double* generalAlgorithmSortedMatrix(CondMatrix* triangularMatrix, void* rhs, int solveType, int triangularForm, bool rhsIsCondMatrix){
	//what if L is 0 ? You divide by zeros ? => try a simple matrix in python with
 	//this problem to see how it deal with it !
	
	//read rhs to 'array'
	double* solution = (double*) malloc(sizeof(double)*(triangularMatrix->nRow));

	if(solution == NULL){
		fprintf(stderr,"Error while allocating the solution in general algorithm.");
		return NULL;
	}

	//initialiazing the solution to the rhs
	//two possibilities either travel two times or put an if in it to put a value of rhs or zeros
	int64_t i, indexRhs = 0, nextRhsVal;
	
	if(rhsIsCondMatrix){
	
		CondMatrix* rhsCond = (CondMatrix*) rhs;
		if(rhsCond->indices != NULL){
			nextRhsVal = rhsCond->indices[indexRhs];
		}
		else{
			nextRhsVal = rhsCond->nRow;
		}
		
		for(i = 0; i < rhsCond->nRow; i++){

			if(i == nextRhsVal){
				solution[i] = rhsCond->data[indexRhs];
				indexRhs++;
				if(indexRhs < rhsCond->nnz){
					nextRhsVal = rhsCond->indices[indexRhs];
				}
			}
			else{
				solution[i] = 0;
			}
		}
	}
	else{
		double* rhsArray = (double*) rhs;

		//intialize the solution to the rhs
		for(i = 0; i < triangularMatrix->nRow; i++){
			solution[i] = rhsArray[i];
		}
	}

	int64_t k = 0;
	double valMatrix, solution_i;


	if(solveType == triangularForm){
		for(i = 0; i < triangularMatrix->nRow; i++){
			//first element of the corresponding col/ row
			valMatrix = triangularMatrix->data[triangularMatrix->indPtr[i]];

			if( valMatrix != 0){
				solution[i] = solution[i]/valMatrix;
			}

			solution_i = solution[i];

			//modifying the value impacted by solution_i
			for(k = triangularMatrix->indPtr[i] + 1; k < triangularMatrix->indPtr[i+1]; k++){

				solution[triangularMatrix->indices[k]] = solution[triangularMatrix->indices[k]] - solution_i*triangularMatrix->data[k];

			}

		}
	}
	else{
	
		for(i = triangularMatrix->nRow - 1; i >= 0; i--){
			//first element of the corresponding col/ row
			valMatrix = triangularMatrix->data[triangularMatrix->indPtr[i+1] - 1];
			
			if(valMatrix != 0){
				solution[i] = solution[i]/valMatrix;
			}

			solution_i = solution[i];

			//modifying the value impacted by solution_i
			for(k = triangularMatrix->indPtr[i]; k < triangularMatrix->indPtr[i+1] - 1; k++){
				
				solution[triangularMatrix->indices[k]] = solution[triangularMatrix->indices[k]] - solution_i*triangularMatrix->data[k];

			}

		}
	}

	return solution;
}

double* generalAlgorithm(CondMatrix* triangularMatrix, void* rhs, int solveType, int triangularForm, bool rhsIsCondMatrix){
	
	double* solution;

	bool secondCondition = true;

	if(rhsIsCondMatrix){

		CondMatrix* rhsCond = (CondMatrix*) rhs;

		if(!rhsCond->indicesSorted){
			secondCondition = false;
		}

	}
	
	if(triangularMatrix->indicesSorted && secondCondition){
		solution = generalAlgorithmSortedMatrix(triangularMatrix, rhs, solveType, triangularForm, rhsIsCondMatrix);
	}
	else{
		solution = generalAlgorithmNotSortedMatrix(triangularMatrix, rhs, solveType, triangularForm, rhsIsCondMatrix);
	}

	return solution;
}

