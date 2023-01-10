#ifndef ALGO_H
#define ALGO_H

#include "mtx.h"

struct ListElement{
	int64_t index;
	double data;
	struct ListElement* next;
};

typedef struct ListElement ListElement;

double* generalAlgorithm(CondMatrix* triangularMatrix, void* rhs, int solveType, int isLowerTriangular, bool rhsIsCondMatrix);

CondMatrix* onePhaseAlgorithm(CondMatrix* triangularMatrix, CondMatrix* rhs, int solveType, int isLowerTriangular);

CondMatrix* twoPhasesAlgorithm(CondMatrix* triangularMatrix, CondMatrix* rhs, int solveType, int isLowerTriangular);

void freeListElement(ListElement* listNode);

#endif