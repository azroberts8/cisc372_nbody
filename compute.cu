#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>

__global__ void calcAccel(vector3** d_accels, vector3* d_hPos, double* d_mass) {
	int i = (blockIdx.y * blockDim.y) + threadIdx.y;
	int j = (blockIdx.x * blockDim.x) + threadIdx.x;
	int k = threadIdx.z;
	if(i==j) {
		FILL_VECTOR(d_accels[i][j], 0, 0, 0);
	} else {
		vector3 distance;
		//for(int k = 0; k < 3; k++) distance[k] = d_hPos[i][k] - d_hPos[j][k];
		distance[k] = d_hPos[i][k] - d_hPos[j][k];
		double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
		double magnitude=sqrt(magnitude_sq);
		double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
		FILL_VECTOR(d_accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	}
}

__global__ void sumAccel(vector3** d_accels, vector3* d_hVel, vector3* d_hPos) {
	int i = (blockIdx.y * blockDim.y) + threadIdx.y;
	int j = (blockIdx.x * blockDim.x) + threadIdx.x;
	int k = threadIdx.z;

	__shared__ vector3 d_sum[NUMENTITIES];

	d_sum[i / NUMENTITIES][k]+=d_accels[i][j][k];
	d_hVel[i][k]+=d_sum[i / NUMENTITIES][k]*INTERVAL;
	d_hPos[i][k]+=d_hVel[i][k]*INTERVAL;
}

int calcGridDim (int block_width, int entity_count) {
	if (entity_count < block_width) return 1;
    int grid_width = entity_count / block_width;
    if (entity_count % block_width == 0) return grid_width;
    return grid_width + 1;
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	//int i,j,k;
	//first compute the pairwise accelerations.  Effect is on the first argument.
	// for (i=0;i<NUMENTITIES;i++){
	// 	for (j=0;j<NUMENTITIES;j++){
	// 		if (i==j) {
	// 			FILL_VECTOR(accels[i][j],0,0,0);
	// 		}
	// 		else{
	// 			vector3 distance;
	// 			for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
	// 			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
	// 			double magnitude=sqrt(magnitude_sq);
	// 			double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
	// 			FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	// 		}
	// 	}
	// }

	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) {
		printf("Pre Err: %s\n", cudaGetErrorString(err));
	}

	int gridWidth = calcGridDim(16, NUMENTITIES);
	dim3 accelGridDim (gridWidth, gridWidth, 1);
	dim3 accelBlockDim (16, 16, 3);
	calcAccel<<<accelGridDim, accelBlockDim>>>(d_accels, d_hPos, d_mass);

	err = cudaGetLastError();
	if(cudaSuccess != err) {
		printf("Cuda Err: %s\n", cudaGetErrorString(err));
	}

	cudaDeviceSynchronize();

	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.

	// sumAccel<<<accelGridDim, accelBlockDim>>>(d_accels, d_hVel, d_hPos);
	// cudaDeviceSynchronize();
	
	
	// for (i=0;i<NUMENTITIES;i++){
	// 	vector3 accel_sum={0,0,0};
	// 	for (j=0;j<NUMENTITIES;j++){
	// 		for (k=0;k<3;k++)
	// 			accel_sum[k]+=accels[i][j][k];
	// 	}
	// 	//compute the new velocity based on the acceleration and time interval
	// 	//compute the new position based on the velocity and time interval
	// 	for (k=0;k<3;k++){
	// 		hVel[i][k]+=accel_sum[k]*INTERVAL;
	// 		hPos[i][k]+=hVel[i][k]*INTERVAL;
	// 	}
	// }
}
