#include "../headers/dynamic_array.h"

ItemArray DynamicArrayGetAt(DynamicArray* array, int64_t indexInArray){
	//to be faster nothing is checked
	return array->internalArray[indexInArray];
}

void DynamicArraySetAt(DynamicArray* array, int64_t indexInArray, ItemArray item){
	
	if(array->capacity <= indexInArray){

		//compute the new size
		double newSize = array->capacity * pow(2, ceil(log2(ceil( (double) (indexInArray+1)/array->capacity))));
		printf("New size = %lf, capacity = %lld, index = %lld\n", newSize, array->capacity, indexInArray);
		array->internalArray = realloc(array->internalArray, newSize*sizeof(ItemArray));
		
		//set element
		array->internalArray[indexInArray] = item;
		array->capacity = (int64_t) newSize;

	}
	else{
		array->internalArray[indexInArray] = item;
	}
}

DynamicArray* allocDynamicArray(int64_t capacity){
	
	DynamicArray* array = (DynamicArray*) malloc(sizeof(DynamicArray*));
	
	if(array == NULL){
		return NULL;
	}

	array->internalArray = (ItemArray*) malloc(sizeof(ItemArray)*capacity);

	if(array->internalArray == NULL){
		free(array);
		return NULL;
	}
	array->capacity = capacity;
}

void freeDynamicArray(DynamicArray* array){

	if(array != NULL){
		if(array->internalArray != NULL){
			free(array->internalArray);
		}
		free(array);
	}
}