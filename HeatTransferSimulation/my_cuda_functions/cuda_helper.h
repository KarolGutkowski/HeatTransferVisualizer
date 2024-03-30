#pragma once
#include <stdio.h>
#include <driver_types.h>
// this is my attempt at making some cuda helper macros
// another weak attempt at copying what great minds at nvidia have created
// honestly im doing this to understand how to make those things myself in the future
// hopefully one day some kid will be doing the same with my code once i work at nvidia :)

template <typename T>
void check_errors(T result, char const* func, char const* file, int const line) {
	if (result) {
		fprintf(stderr, "CUDA error from function call: %s\nerror code=%d(%s)\nfile: %s\nline: %d\n",
			func, static_cast<unsigned>(result), cudaGetErrorName(result), file, line);
	}
}

#define checkCudaErrors(val) check_errors(val, #val, __FILE__, __LINE__)