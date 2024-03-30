#include "gpu_calculation.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "my_cuda_functions/cuda_helper.h"
#include "math.h"

__device__ uchar3 calculate_rgb_from_temp_in_range(float temp, float min_temp, float max_temp);

__global__ void calculate_heat_equation(const float alpha, const float delta_t, 
	const float delta_x, const float delta_y,
	uchar3* pixels, const float* temps, float* temps_after,
	int width, int height)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if (row >= height || col >= width)
		return;

	int idx = row * width + col;

	// boundary conditions of differential equation
	if (row > 0 && row < height-1 && col > 0 && col < width-1)
	{
		temps_after[idx] = alpha * delta_t / (delta_x * delta_x) * (
			temps[idx + 1] - 4 * temps[idx] + temps[idx - 1] + temps[idx + width] + temps[idx - width]
			) + temps[idx];
	}

	pixels[idx] = calculate_rgb_from_temp_in_range(temps[idx], 0.0f, 3000.0f);
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

	dim3 blockDim = dim3(32, 32);
	dim3 gridDim = dim3((width + 32 - 1) / 32, (height + 32 - 1) / 32);
	calculate_heat_equation << <gridDim, dim3(32, 32) >> > (
		alpha, delta_t, 
		delta_x, delta_y,
		pixels, temps_d, temps_after,
		width, height);

	checkCudaErrors(cudaDeviceSynchronize());

	cudaMemcpy(temps, temps_after, temps_size, cudaMemcpyDeviceToHost);
	cudaFree(temps_d);
	cudaFree(temps_after);
}