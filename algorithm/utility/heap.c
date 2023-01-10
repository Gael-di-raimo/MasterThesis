#include "../headers/heap.h"

static void swapItem(Heap* heap, int64_t index1, int64_t index2){
	
	HeapNode tmp = heap->array[index1];
	heap->array[index1] = heap->array[index2];
	heap->array[index2] = tmp;
	
}

static void maxHeapifyDown(Heap* heap, int64_t index){

	int64_t	leftIndex = 2*index + 1;

	if(leftIndex < heap->sizeHeap){

		int64_t rightIndex = 2*index + 2;
		
		int64_t mustBeParentIndex = index;

		if(heap->array[leftIndex].key > heap->array[index].key){
			mustBeParentIndex = leftIndex;
		}

		if(rightIndex < heap->sizeHeap && heap->array[rightIndex].key > heap->array[mustBeParentIndex].key){
			mustBeParentIndex = rightIndex;
		}

		if(mustBeParentIndex != index){
			swapItem(heap, mustBeParentIndex, index);
			maxHeapifyDown(heap, mustBeParentIndex);
		}

	}
	
}

static void minHeapifyDown(Heap* heap, int64_t index){

	int64_t	leftIndex = 2*index + 1;

	if(leftIndex < heap->sizeHeap){

		int64_t rightIndex = 2*index + 2;
		
		int64_t mustBeParentIndex = index;

		if(heap->array[leftIndex].key < heap->array[index].key){
			mustBeParentIndex = leftIndex;
		}

		if(rightIndex < heap->sizeHeap && heap->array[rightIndex].key < heap->array[mustBeParentIndex].key){
			mustBeParentIndex = rightIndex;
		}

		if(mustBeParentIndex != index){
			swapItem(heap, mustBeParentIndex, index);
			minHeapifyDown(heap, mustBeParentIndex);
		}

	}
	
}

static void maxHeapifyUp(Heap* heap, int64_t index){

	int64_t parentIndex = (index-1)/2;

	while(heap->array[index].key > heap->array[parentIndex].key){

		swapItem(heap, index, parentIndex);
		index = parentIndex;
		parentIndex = (index-1)/2;
	
	}
}

static void minHeapifyUp(Heap* heap, int64_t index){

	int64_t parentIndex = (index-1)/2;

	while(heap->array[index].key < heap->array[parentIndex].key){

		swapItem(heap, index, parentIndex);
		index = parentIndex;
		parentIndex = (index-1)/2;
	
	}
}

int addMaxHeapItem(Heap* heap, int64_t key, double data){

	int64_t sizeHeap = heap->sizeHeap;

	if(sizeHeap == heap->capacityHeap){

		//not enough space so double the size of the array
		heap->array = realloc(heap->array, heap->capacityHeap*sizeof(HeapNode)*2);

		if(heap->array == NULL){
			//failed realloc
			return 1;
		}
		
		heap->capacityHeap = heap->capacityHeap*2;
	}

	heap->array[sizeHeap].key = key;
	heap->array[sizeHeap].data = data;
	heap->sizeHeap++;

	maxHeapifyUp(heap, sizeHeap);

	return 0;
}

int addMinHeapItem(Heap* heap, int64_t key, double data){

	int64_t sizeHeap = heap->sizeHeap;

	if(sizeHeap == heap->capacityHeap){

		//not enough space so double the size of the array
		heap->array = realloc(heap->array, heap->capacityHeap*sizeof(HeapNode)*2);

		if(heap->array == NULL){
			//failed realloc
			return 1;
		}
		
		heap->capacityHeap = heap->capacityHeap*2;
	}

	heap->array[sizeHeap].key = key;
	heap->array[sizeHeap].data = data;
	heap->sizeHeap++;

	minHeapifyUp(heap, sizeHeap);

	return 0;
}

HeapNode* remTopMaxHeapItem(Heap* heap){

	if(heap->sizeHeap == 0){
		return NULL;
	}

	//saving the top of the heap node
	HeapNode* retNode = (HeapNode*) malloc(sizeof(HeapNode));
	retNode->key = heap->array[0].key;
	retNode->data =  heap->array[0].data;

	//swap the top of the heap with the last leaf
	swapItem(heap, 0, heap->sizeHeap-1);
	

	//removing the item
	heap->array[heap->sizeHeap - 1].key = -INF_KEY;
	heap->sizeHeap--;

	//reorder the heap
	maxHeapifyDown(heap, 0);

	return retNode;
}

HeapNode* remTopMinHeapItem(Heap* heap){

	if(heap->sizeHeap == 0){
		return NULL;
	}

	//saving the top of the heap node
	HeapNode* retNode = (HeapNode*) malloc(sizeof(HeapNode));
	retNode->key = heap->array[0].key;
	retNode->data =  heap->array[0].data;

	//swap the top of the heap with the last leaf
	swapItem(heap, 0, heap->sizeHeap-1);
	

	//removing the item
	heap->array[heap->sizeHeap - 1].key = INF_KEY;
	heap->sizeHeap--;

	//reorder the heap
	minHeapifyDown(heap, 0);

	return retNode;
}


Heap* buildMaxHeapFromArray(int64_t* keys, double* data, int64_t arraySize){
	
	Heap* heap = (Heap*) malloc(sizeof(Heap));

	if(heap == NULL){
		return NULL;
	}

	//copy the array into dynamic array
	heap->array = (HeapNode*) malloc(sizeof(HeapNode)*arraySize);

	if(heap->array == NULL){
		free(heap);
		return NULL;
	}

	//call heapify down on each node
	int64_t i;

	heap->sizeHeap = arraySize;
	heap->capacityHeap = arraySize;

	for(i = 0; i < arraySize; i++){
		heap->array[i].key = keys[i];
		heap->array[i].data = data[i];
	}

	for(i = arraySize - 1; i >= 0; i--){
		maxHeapifyDown(heap, i);
	}
	
	return heap;
}

Heap* buildMinHeapFromArray(int64_t* keys, double* data, int64_t arraySize){
	
	Heap* heap = (Heap*) malloc(sizeof(Heap));

	if(heap == NULL){
		return NULL;
	}

	//copy the array into dynamic array
	heap->array = (HeapNode*) malloc(sizeof(HeapNode)*arraySize);

	if(heap->array == NULL){
		free(heap);
		return NULL;
	}

	//call heapify down on each node
	int64_t i;

	heap->sizeHeap = arraySize;
	heap->capacityHeap = arraySize;

	for(i = 0; i < arraySize; i++){
		heap->array[i].key = keys[i];
		heap->array[i].data = data[i];
	}

	for(i = arraySize - 1; i >= 0; i--){
		minHeapifyDown(heap, i);
	}
	
	return heap;
}

void freeHeap(Heap* heap){

	if(heap != NULL){

		if(heap->array != NULL){
			free(heap->array);
		}

		free(heap);
	}
}
