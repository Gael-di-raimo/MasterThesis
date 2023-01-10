#ifndef FEATURES_H
#define FEATURES_H

#include <stdint.h>
#include <math.h>
#include "mtx.h"

int64_t f0(CondMatrix* triangularMatrix, CondMatrix* rhs, float prunePercentage, int solveType, int isLowerTriangular);
int64_t f1(CondMatrix* triangularMatrix, CondMatrix* rhs, float prunePercentage, int solveType, int isLowerTriangular);
int64_t f2(CondMatrix* triangularMatrix, CondMatrix* rhs, float prunePercentage, int solveType, int isLowerTriangular);
double f3(CondMatrix* triangularMatrix, CondMatrix* rhs, float prunePercentage, int solveType, int isLowerTriangular);

#endif