#ifndef  GPU_CALCULATION
#define GPU_CALCULATION

#include "vector_types.h"

void calculate_heat_equation_at_time(
	float delta_t, float delta_x, float delta_y, float alpha, 
	uchar3* pixels, float* temps, int width, int height);
#endif // ! GPU_CALCULATION