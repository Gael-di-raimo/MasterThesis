#ifndef MTX_H
#define MTX_H

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

struct CondMatrix{
	int64_t* indPtr;
	int64_t* indices;
	double* data;
	int64_t nnz;
	int64_t nCol;
	int64_t nRow;
	bool format;//1 = csc and 0 = csr.
	bool indicesSorted;
};

typedef struct CondMatrix CondMatrix;

CondMatrix* ReadMatrix(char* path, bool format, bool indicesSorted);

void printCondMatrix(CondMatrix* matrix);

int fprintCondMatrix(CondMatrix* matrix, char* path);

int WriteMatrix(char* path, CondMatrix* matrix);

void freeCondMatrix(CondMatrix* matrix);

CondMatrix* allocateCondMatrix(int64_t nnz, int64_t nCol, int64_t nRow, bool format);

#endif