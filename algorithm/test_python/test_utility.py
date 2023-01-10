import os


if __name__ == "__main__":
	testDynamic = False
	testHeap = True
	if testDynamic:
		if os.path.isfile("test.exe"):
			os.system("del test.exe")

		os.system("gcc  ../testUtility.c ../utility/dynamic_array.c ../headers/dynamic_array.h -o test")
		if os.path.isfile("test.exe"):
			os.system("test")

	if testHeap:

		if os.path.isfile("test.exe"):
			os.system("del test.exe")

		os.system("gcc  ../testHeap.c ../utility/heap.c ../headers/heap.h -o test")
		
		if os.path.isfile("test.exe"):
			os.system("test")