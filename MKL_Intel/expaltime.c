/* C source code is found in dgemm_example.c */
// compiled with icc gemm.c -Wl,-rpath,/opt/intel/compilers_and_libraries_2017.4.181/mac/mkl/lib  -Wl,-rpath,/opt/intel/compilers_and_libraries_2017.4.181/mac/compiler/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -qopenmp -mkl -liomp5
//icc expaltime.c -Wl,-rpath,/opt/intel/compilers_and_libraries_2017.4.181/mac/mkl/lib  -Wl,-rpath,/opt/intel/compilers_and_libraries_2017.4.181/mac/compiler/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -qopenmp -mkl -liomp5 -parallel
//Henrys-MacBook-Pro:desktop henryboswell$ ./a.out

//source http://www.mscs.dal.ca/cluster/manuals/intel-mkl/examples/vmlc/source/
#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>

#define VEC_LEN 200000



#define __SEXP_BEG -17.0
#define __SEXP_END 18.0

double get_time() //https://stackoverflow.com/questions/2349776/how-can-i-benchmark-c-code-easily
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}

int main()
{
    
    omp_set_num_threads(1);
   float fA[VEC_LEN],fB1[VEC_LEN],fB2[VEC_LEN];
    double s_initial, s_elapsed;

    int i=0,vec_len=VEC_LEN;
    double CurRMS,MaxRMS=0.0;

    for(i=0;i<vec_len;i++) {
        fA[i]= rand() % 20;
        fB1[i]=0.0;
        
    }
    
    
    
    vmlSetMode( VML_EP );
 
    
     double start =get_time();
    
    vsExp(vec_len,fA,fB1);
    
    double bench = (get_time() - start);
    printf("Time = %f \n", bench * 1000);
    
    
    
    
  
    
    
    for(i=0;i<vec_len;i++) {
        fA[i]= rand() % 20;
        fB1[i]=0.0;
        
    }
   start =get_time();
    
    for(i=0;i<vec_len;i++) {
        
        fB2[i]=(float)exp(fA[i]);
        
    }
    
     bench = (get_time() - start);
    printf("Time = %f \n", bench*1000);

    
    
    /*
    
  printf("vsExp test/example program\n\n");
  printf("           Argument                     vsExp                      Expf\n");
  printf("===============================================================================\n");
  for(i=0;i<vec_len;i++) {
    printf("% 25.14f % 25.14e % 25.14e\n",fA[i],fB1[i],fB2[i]);
    CurRMS=(fB1[i]-fB2[i])/(0.5*(fB1[i]+fB2[i]));
    if(MaxRMS<CurRMS) MaxRMS=CurRMS;
  }
  printf("\n");
  printf("Maximum relative error: %.2f\n",MaxRMS);
     */
    

  return 0;

}
