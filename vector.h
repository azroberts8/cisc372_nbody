#ifndef __TYPES_H__
#define __TYPES_H__

typedef double vector3[3];
#define FILL_VECTOR(vector,a,b,c) {vector[0]=a;vector[1]=b;vector[2]=c;}
extern vector3 *hVel, *d_hVel;
extern vector3 *hPos, *d_hPos;
extern vector3** h_accels;
extern vector3** d_accels;
extern double* d_mass;
extern double *mass;

#endif