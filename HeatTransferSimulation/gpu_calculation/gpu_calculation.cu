#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "gpu_calculation.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "my_cuda_functions/cuda_helper.h"
#include "math.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2*1)

__device__ uchar3 calculate_rgb_from_temp_in_range(float temp, float min_temp, float max_temp);

__global__ void calculate_heat_equation(const float alpha, const float delta_t, 
	const float delta_x, const float delta_y,
	uchar3* pixels, const float* temps, float* temps_after,
	const int width, const int height)
{
	int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
	int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

	__shared__ float temps_s[IN_TILE_DIM][IN_TILE_DIM];

	// load data into shared memory
	// row >= 0 and row < height to protect from accessing not exisiting elements of temps array
	// similiarly for col variable
	// temps_s is size of input tile so mapping is the same as threads in the block
	if (row >= 0 && row < height &&
		col >= 0 && col < width)
	{
		temps_s[threadIdx.y][threadIdx.x] = temps[row * width + col];
	}

	__syncthreads();
	

	// boundary conditions of differential equation
	if (row >= 1 && row < height -1 &&
		col >= 1 && col < width - 1 )
	{
		if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
			threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1)
		{
			temps_after[row * width + col] = (alpha * delta_t) / (delta_x * delta_x) *
				(
					temps_s[threadIdx.y][threadIdx.x + 1] - 4 * temps_s[threadIdx.y][threadIdx.x] +
					temps_s[threadIdx.y][threadIdx.x - 1] + temps_s[threadIdx.y - 1][threadIdx.x] +
					temps_s[threadIdx.y + 1][threadIdx.x]
					) + temps_s[threadIdx.y][threadIdx.x];
		}
	}


	if (row >= 0 && row < height &&
		col >= 0 && col < width)
	{
		pixels[row * width + col] = calculate_rgb_from_temp_in_range(temps[row * width + col],0.0f, 3000.0f);
	}
}

__device__ uchar3 calculate_rgb_from_temp_in_range(float temp, float min_temp, float max_temp)
{
	uchar3 temperature;
	float ratio = (temp - min_temp) / (max_temp - min_temp);

	if (ratio >= 0.5f)
	{
		temperature.x = 255;
		temperature.y = (1 - ratio) / 0.5f * 255;
		temperature.z = 0;
	}
	else
	{
		temperature.x = 0;
		temperature.y = (ratio / 0.5f) * 255;
		temperature.z = 255;
	}
	return temperature;
}


void calculate_heat_equation_at_time(float delta_t, float delta_x,
	float delta_y, float alpha, uchar3* pixels, float* temps, int width, int height)
{
	float* temps_d;
	size_t temps_size = sizeof(float) * width * height;
	cudaMalloc((void**)&temps_d, temps_size);
	cudaMemcpy(temps_d, temps, temps_size, cudaMemcpyHostToDevice);

	float* temps_after;
	cudaMalloc((void**)&temps_after, temps_size);
	// copying cause i want to prevail boundary conditions
	// another approach would be setting them in the kernel every time
	cudaMemcpy(temps_after, temps, temps_size, cudaMemcpyHostToDevice);

	dim3 blockDim = dim3(IN_TILE_DIM, IN_TILE_DIM);
	dim3 gridDim = dim3((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
	calculate_heat_equation << <gridDim, blockDim >> > (
		alpha, delta_t, 
		delta_x, delta_y,
		pixels, temps_d, temps_after,
		width, height);

	checkCudaErrors(cudaDeviceSynchronize());

	cudaMemcpy(temps, temps_after, temps_size, cudaMemcpyDeviceToHost);
	cudaFree(temps_d);
	cudaFree(temps_after);
}