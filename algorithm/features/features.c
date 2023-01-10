#include "../headers/features.h"
#include "../headers/heap.h"


int64_t f0(CondMatrix* triangularMatrix, CondMatrix* rhs, float prunePercentage, int solveType, int isLowerTriangular){
	
	int64_t res = 0, i, row_index;


	if(solveType == isLowerTriangular){

		for(i = 0; i < (int64_t) rhs->nnz*prunePercentage; i++){
			row_index = rhs->indices[i];
			res = res + triangularMatrix->indPtr[row_index+1] - triangularMatrix->indPtr[row_index];
		}

	}
	else{
		for(i = rhs->nnz*(1-prunePercentage); i < rhs->nnz; i++){
			row_index = rhs->indices[i];
			res = res + triangularMatrix->indPtr[row_index+1] - triangularMatrix->indPtr[row_index];
		}
	}

	return res;

}

int64_t f1(CondMatrix* triangularMatrix, CondMatrix* rhs, float prunePercentage, int solveType, int isLowerTriangular){

	int64_t res = 0, i;

	if(solveType == isLowerTriangular){

		for(i = 0; i < (int64_t) rhs->nnz*prunePercentage; i++){
			res = res + (rhs->nRow - rhs->indices[i]);
		}

	}
	else{

		for(i = rhs->nnz*(1-prunePercentage); i < rhs->nnz; i++){
			res = res + (rhs->nRow - rhs->indices[i]);
		}
	}

	return res;
}

int64_t f2(CondMatrix* triangularMatrix, CondMatrix* rhs, float prunePercentage, int solveType, int isLowerTriangular){

	int64_t res = 0, i;

	if(solveType == isLowerTriangular){

		for(i = 0; i < (int64_t) rhs->nnz*prunePercentage; i++){
			res = res + (rhs->nRow - rhs->indices[i])*(triangularMatrix->indPtr[i+1] - triangularMatrix->indPtr[i]);
		}

	}
	else{

		for(i = rhs->nnz*(1-prunePercentage); i < rhs->nnz; i++){
			res = res + (rhs->nRow - rhs->indices[i])*(triangularMatrix->indPtr[i+1] - triangularMatrix->indPtr[i]);
		}
	}

	return res;
}

double f3(CondMatrix* triangularMatrix, CondMatrix* rhs, float prunePercentage, int solveType, int isLowerTriangular){

	double res = 0;
	int64_t i;

	if(solveType == isLowerTriangular){

		for(i = 0; i < (int64_t) rhs->nnz*prunePercentage; i++){
			res = res + pow( (double) (rhs->nRow - rhs->indices[i]), (double) (triangularMatrix->indPtr[i+1] - triangularMatrix->indPtr[i]));
		}

	}
	else{

		for(i = (int64_t) rhs->nnz*(1-prunePercentage); i < rhs->nnz; i++){
			res = res + pow( (double) (rhs->nRow - rhs->indices[i]), (double) (triangularMatrix->indPtr[i+1] - triangularMatrix->indPtr[i]));
		}
	}

	return res;
}
