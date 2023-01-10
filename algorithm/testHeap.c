#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <stdbool.h>
#include <time.h>
#include "headers/heap.h"

int main(int argc, char *argv[]) {
	/*
	//create a heap with some data
	printf("Starting heap test\n");
	int64_t keys[12] = {9,5,6,4,3,2,1,8,7,0,10,11}; 
	double values[12] = {0,0,0,0,0,0,0,0,0,0,0,0};

	Heap* heap =  buildHeapFromArray(keys, values, 12);
	if(heap == NULL){
		printf("Error while allocating the heap.\n");
	}
	printf("SizeHeap = %lld\n", heap->sizeHeap);
	printf("CapacityHeap = %lld\n", heap->capacityHeap);

	//add some items
	
	int res = addHeapItem(heap, -1, 1);
	res = addHeapItem(heap, 13, 1);
	printf("SizeHeap = %lld\n", heap->sizeHeap);
	printf("CapacityHeap = %lld\n", heap->capacityHeap);
	
	//pop the minimal values util nothing is left ?
	int i;
	HeapNode* node = remTopHeapItem(heap);
	
	while(node != NULL){
		printf("NodeKey = %lld, NodeData = %lf\n", node->key, node->data);
		node = remTopHeapItem(heap);
	}
	printf("End heap test\n");
	*/

	//create a heap with some data
	printf("Starting heap test\n");
	int64_t keys[12] = {2,2,2,2,2,2,1,8,7,0,10,11}; 
	double values[12] = {0,1,2,3,4,5,0,0,0,0,0,0};

	Heap* heap =  buildHeapFromArray(keys, values, 12);
	if(heap == NULL){
		printf("Error while allocating the heap.\n");
	}
	printf("SizeHeap = %lld\n", heap->sizeHeap);
	printf("CapacityHeap = %lld\n", heap->capacityHeap);

	//add some items
	
	int res = addHeapItem(heap, -1, 1);
	res = addHeapItem(heap, 1, 1);
	res = addHeapItem(heap, 2, 6);
	printf("SizeHeap = %lld\n", heap->sizeHeap);
	printf("CapacityHeap = %lld\n", heap->capacityHeap);
	
	//pop the minimal values util nothing is left ?
	int i;
	HeapNode* node = remTopHeapItem(heap);
	
	while(node != NULL){
		printf("NodeKey = %lld, NodeData = %lf\n", node->key, node->data);
		node = remTopHeapItem(heap);
	}
	printf("End heap test\n");
	return 0;


}