#ifndef DYNAMIC_ARR_H
#define DYNAMIC_ARR_H

#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
// The code is inspired from https://brilliant.org/wiki/dynamic-arrays/

typedef struct{
	int64_t index;
	double data;
}ItemArray;

typedef struct {
	int64_t capacity;
	ItemArray* internalArray;
}DynamicArray;

DynamicArray* allocDynamicArray(int64_t capacity);

ItemArray DynamicArrayGetAt(DynamicArray* array, int64_t indexInArray);

void DynamicArraySetAt(DynamicArray* array, int64_t indexInArray, ItemArray item);

void freeDynamicArray(DynamicArray* array);

#endif