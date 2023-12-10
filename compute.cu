#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "vector.h"
#include "config.h"

__global__ void calcAccel(vector3** d_accels, vector3* d_hPos, double* d_mass) {
	int task = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(task >= NUMENTITIES * NUMENTITIES) return;
	int i = task / NUMENTITIES;
	int j = task % NUMENTITIES;

	if(i == j) {
		FILL_VECTOR(d_accels[i][j], 0, 0, 0);
	} else {
		vector3 distance;
		for(int k = 0; k < 3; k++) distance[k] = d_hPos[i][k] - d_hPos[j][k];
		double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
		double magnitude = sqrt(magnitude_sq);
		double accelmag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;
		FILL_VECTOR(d_accels[i][j], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
	}
}

__global__ void sumAccels(vector3** d_accels, vector3* d_hVel, vector3* d_hPos) {
	int i, j, k;
	i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i >= NUMENTITIES) return;

	vector3 accel_sum = {0, 0, 0};
	for(j = 0; j < NUMENTITIES; j++) {
		for(k=0;k<3;k++) accel_sum[k] += d_accels[i][j][k];
	}

	for(k = 0; k < 3; k++) {
		d_hVel[i][k] += accel_sum[k] * INTERVAL;
		d_hPos[i][k] += d_hVel[i][k] * INTERVAL;
	}
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(int blocks, int threads){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	//int i,j,k;
	//first compute the pairwise accelerations.  Effect is on the first argument.
	// for (i=0;i<NUMENTITIES;i++){
	// 	for (j=0;j<NUMENTITIES;j++){
	// 		if (i==j) {
	// 			FILL_VECTOR(h_accels[i][j],0,0,0);
	// 		}
	// 		else{
	// 			vector3 distance;
	// 			for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
	// 			double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
	// 			double magnitude=sqrt(magnitude_sq);
	// 			double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
	// 			FILL_VECTOR(h_accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	// 		}
	// 	}
	// }

	calcAccel<<<blocks, threads>>>(d_accels, d_hPos, d_mass);
	cudaDeviceSynchronize();

	blocks = ceil((float)NUMENTITIES / 1024);
	threads = ceil((float)NUMENTITIES / blocks);
	sumAccels<<<blocks, threads>>>(d_accels, d_hVel, d_hPos);
	cudaDeviceSynchronize();



	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	// for (i=0;i<NUMENTITIES;i++){
	// 	vector3 accel_sum={0,0,0};
	// 	for (j=0;j<NUMENTITIES;j++){
	// 		for (k=0;k<3;k++)
	// 			accel_sum[k]+=h_accels[i][j][k];
	// 	}
	// 	//compute the new velocity based on the acceleration and time interval
	// 	//compute the new position based on the velocity and time interval
	// 	for (k=0;k<3;k++){
	// 		hVel[i][k]+=accel_sum[k]*INTERVAL;
	// 		hPos[i][k]+=hVel[i][k]*INTERVAL;
	// 	}
	// }
}
