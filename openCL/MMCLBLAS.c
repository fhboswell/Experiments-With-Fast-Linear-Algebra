//
//  main.c
//  Martix
//
//  Created by Franklin Henry Boswell on 7/20/17.
//  Copyright © 2017 Franklin Henry Boswell. All rights reserved.
//


//compile with  gcc MMCLBLAS.c -lclblas -framework OpenCL

#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <pthread.h>


#include <clBLAS.h>

// Set the alpha and beta values for the cuBLAS and clBlas libraries. Note that the myGEMM kernels
// for simplicity only support alpha values of 1 and beta values of 0.
#define ALPHA 1.0f
#define BETA 0.0f



void printMAT(float *MAT,int N){
    
    printf("N = %d \n", N);
    for (int i = 0; i <  N; i++){
        printf("\n");
        for (int j = 0; j < N; j++){
            printf(" %f ",*(MAT + i*N + j));
            
        }
    }
}






void naive_IKJ_Square(float *MAT1, float *MAT2, float*MAT4, int N){
    
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                // MAT3[i][j] += MAT1[i][k] * MAT2[k][j];
                *(MAT4 + i*N + j) += *(MAT1 + i*N + k) * *(MAT2 + k*N + j);
            }
        }
    }
    
    
}

void clblasMM(float* A, float* B, float* C,
               int K, int M, int N) {
    cl_int err;
    
    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;
    
    char* value;
    size_t valueSize;
        clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
    
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, 0, NULL, &valueSize);
    //value = (char*) malloc(valueSize);
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, valueSize, value, NULL);
    //printf("%d. Device: %s\n", 2+1, value);
    
    cl_context context = clCreateContext(NULL, 1, &devices[2], NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, devices[2], 0, NULL);
    cl_event event = NULL;
    
    err = clblasSetup();
    
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, M*K*sizeof(*A), NULL, &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, K*N*sizeof(*B), NULL, &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, M*N*sizeof(*C), NULL, &err);
    
  
    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M*K*sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K*N*sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
    
    err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans,
                      M, N, K, ALPHA,
                      bufA, 0, M,
                      bufB, 0, K, BETA,
                      bufC, 0, M,
                      1, &queue, 0, NULL, &event);
    
    
    err = clWaitForEvents(1, &event);
    
    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M*N*sizeof(*C), C, 0, NULL, NULL);
    
    // Free the GPU memory objects
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

// =================================================================================================

double get_time() //https://stackoverflow.com/questions/2349776/how-can-i-benchmark-c-code-easily
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}
void proof(){
    
    int N = 128;
    
    float *MAT1 = (float *)malloc(N * N * sizeof(float));
    float *MAT2 = (float *)malloc(N * N * sizeof(float));
    
    
    
    int cutoff;
    if(N < 4100){
        cutoff = 128;
        
    }else{
        cutoff = 256;
    }
    srand((int)time(NULL));
    
    
    for (int i = 0; i <  N; i++)
        for (int j = 0; j < N; j++)
            *(MAT1 + i*N + j) = rand() % 31 + 10;
    
    
    for (int i = 0; i <  N; i++)
        for (int j = 0; j < N; j++)
            *(MAT2 + i*N + j) = rand() % 45;
    
    
    float *MAT = (float *)malloc(N * N * sizeof(float));
    float *MAT4 = (float *)calloc(N * N, sizeof(float));
    float *MAT5 = (float *)calloc(N * N, sizeof(float));
    
    
    
    
    
    naive_IKJ_Square(MAT1, MAT2, MAT, N);
    clblasMM(MAT1, MAT2, MAT4, N, N, N);
    
    
    /*
    printMAT(MAT1, N);
    printMAT(MAT2, N);
    
    printMAT(MAT4, N);
    printMAT(MAT, N);
     */
    
    
    int strassen = 0;
    
    
    for (int i = 0; i <  N; i++){
        for (int j = 0; j < N; j++){
            if(*(MAT4 + i*N + j) !=  *(MAT + i*N + j)){
                printf("fail");
                strassen = 1;
            }
        }
    }
    if(strassen == 0){
        printf("\nSgemm Algorithm output is acurate");
    }else{
        printf("\nSgemm Algorithm output fails");
    }
   
    
    
    
    
    
}
int main(){
    
    int N = 1024;
    
    
    
    
    //printMAT(MAT2, N);
    //printMAT(MAT1, N);
    //square_MAT_Mul_recursion_multi(MAT1, MAT2, MAT4, N, N);
    //naive_IKJ_Square(MAT1, MAT2, MAT4, N);
    
    proof();
    
    printf("\n");
    
    double bench = 0;
    double start =  0;
    
    int smaller_test_sizes[] ={ 32, 64, 128, 256, 512, 1024};
    
    printf("square_MAT_Mul_recursion\n");
    for(int i = 0; i < 6 ; i++){
        N = smaller_test_sizes[i];
        
        float *MAT1 = (float *)malloc(N * N * sizeof(float));
        float *MAT2 = (float *)malloc(N * N * sizeof(float));
        float *MAT4= (float *)calloc(N * N, sizeof(float));
        
        
        int cutoff;
        if(N < 4100){
            cutoff = 128;
            
        }else{
            cutoff = 256;
        }
        srand((int)time(NULL));
        
        
        for (int i = 0; i <  N; i++)
            for (int j = 0; j < N; j++)
                *(MAT1 + i*N + j) = rand() % 31 + 10;
        
        
        for (int i = 0; i <  N; i++)
            for (int j = 0; j < N; j++)
                *(MAT2 + i*N + j) = rand() % 45;
        
        
        
        start =get_time();
        naive_IKJ_Square(MAT1, MAT2, MAT4, N);
        bench = (get_time() - start);
        printf("Size = %d X %d\t", smaller_test_sizes[i],smaller_test_sizes[i]);
        if(i == 0 || i ==1){
            printf("\t");
        }
        printf("Time = %f \n", bench);
        double  Gflops_s = 2.e-9  * N * N * N / bench;
        printf ("Size: %d\tGflop/s: %.3g\n", N, Gflops_s);
        if(i == 0 || i ==1){
            //printf("\t");
        }
        printf("Time = %f \n", bench);
        free(MAT1);
        free(MAT2);
        free(MAT4);
    }

    
    
    int test_sizes[] ={  128, 256, 512, 1024, 2048, 2300,2500,2700, 3000, 3045, 3500};
    
    
    printf("strassen_recursion\n");
    for(int i = 0; i < 11; i++){
        N = test_sizes[i];
        
        float *MAT1 = (float *)malloc(N * N * sizeof(float));
        float *MAT2 = (float *)malloc(N * N * sizeof(float));
        float *MAT4= (float *)calloc(N * N, sizeof(float));
        
        
        int cutoff;
        if(N < 4100){
            cutoff = 128;
            
        }else{
            cutoff = 256;
        }
        srand((int)time(NULL));
        
        
        for (int i = 0; i <  N; i++)
            for (int j = 0; j < N; j++)
                *(MAT1 + i*N + j) = rand() % 31 + 10;
        
        
        for (int i = 0; i <  N; i++)
            for (int j = 0; j < N; j++)
                *(MAT2 + i*N + j) = rand() % 45;
        
        
        
        start =get_time();
        
        clblasMM(MAT1, MAT2, MAT4, N, N, N);
       // strassen_recursion(MAT1, MAT2, MAT4, N, cutoff);
        bench = (get_time() - start);
        printf("Size = %d X %d\t", test_sizes[i],test_sizes[i]);
        
        printf("Time = %f \n", bench*1000);
        double  Gflops_s = 2.e-9  * N * N * N / bench;
        printf ("Size: %d\tGflop/s: %.3g\n", N, Gflops_s);
        if(i == 0 || i ==1){
            //printf("\t");
        }
        printf("Time = %f \n", bench);
        
        free(MAT1);
        free(MAT2);
        free(MAT4);
    }
    
    
   

}



