//
//  main.c
//  Martix
//
//  Created by Franklin Henry Boswell on 7/20/17.
//  Copyright © 2017 Franklin Henry Boswell. All rights reserved.
//

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



void printMAT(double *MAT,int N){
    
    printf("N = %d \n", N);
    for (int i = 0; i <  N; i++){
        printf("\n");
        for (int j = 0; j < N; j++){
            printf(" %f ",*(MAT + i*N + j));
            
        }
    }
}






void naive_IKJ_Square(double *MAT1, double *MAT2, double*MAT4, int N){
    
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                // MAT3[i][j] += MAT1[i][k] * MAT2[k][j];
                *(MAT4 + i*N + j) += *(MAT1 + i*N + k) * *(MAT2 + k*N + j);
            }
        }
    }
    
    
}

extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double
                   *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c,
                   int *ldc );

void blasMM(double* A, double* B, double* C,
               int K, int M, int N) {
    int nn=N;
    int lda=N;
    int ldb=N;
    int ldc=N;
    double alpha=1.0;
    double beta=1.0;
    
    dgemm_( "N", "N", &nn, &nn, &nn,&alpha, B, &lda,
           A,&ldb, &beta, C, &ldc );
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
    
    int N = 4;
    
    double *MAT1 = (double *)malloc(N * N * sizeof(double));
    double *MAT2 = (double *)malloc(N * N * sizeof(double));
    
    
    
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
    
    
    double *MAT = (double *)malloc(N * N * sizeof(double));
    double *MAT4 = (double *)calloc(N * N, sizeof(double));
    double *MAT5 = (double *)calloc(N * N, sizeof(double));
    
    
    
    
    
    naive_IKJ_Square(MAT1, MAT2, MAT, N);
    blasMM(MAT1, MAT2, MAT4, N, N, N);
    
    
    
    printMAT(MAT1, N);
    printMAT(MAT2, N);
    
    printMAT(MAT4, N);
    printMAT(MAT, N);
    
    
    
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
    /*
    double bench = 0;
    double start =  0;
    
    int smaller_test_sizes[] ={ 32, 64, 128, 256, 512, 1024};
    
    printf("square_MAT_Mul_recursion\n");
    for(int i = 0; i < 6 ; i++){
        N = smaller_test_sizes[i];
        
        double *MAT1 = (double *)malloc(N * N * sizeof(double));
        double *MAT2 = (double *)malloc(N * N * sizeof(double));
        double *MAT4= (double *)calloc(N * N, sizeof(double));
        
        
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
        
        double *MAT1 = (double *)malloc(N * N * sizeof(double));
        double *MAT2 = (double *)malloc(N * N * sizeof(double));
        double *MAT4= (double *)calloc(N * N, sizeof(double));
        
        
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
        
        blasMM(MAT1, MAT2, MAT4, N, N, N);
       // strassen_recursion(MAT1, MAT2, MAT4, N, cutoff);
        bench = (get_time() - start);
        printf("Size = %d X %d\t", test_sizes[i],test_sizes[i]);
        
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
    
    */
   

}



