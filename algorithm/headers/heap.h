#ifndef HEAP_H
#define HEAP_H

#define INF_KEY 9223372036854775807

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct{
	int64_t key;
	double data;
}HeapNode;

typedef struct {
	HeapNode* array;
	int64_t sizeHeap;
	int64_t capacityHeap;
}Heap;

Heap* buildMaxHeapFromArray(int64_t* keys, double* data, int64_t sizeHeap);

Heap* buildMinHeapFromArray(int64_t* keys, double* data, int64_t sizeHeap);

int addMaxHeapItem(Heap* heap, int64_t key, double data);
int addMinHeapItem(Heap* heap, int64_t key, double data);

HeapNode* remTopMaxHeapItem(Heap* heap);
HeapNode* remTopMinHeapItem(Heap* heap);

void freeHeap(Heap* heap);

#endif