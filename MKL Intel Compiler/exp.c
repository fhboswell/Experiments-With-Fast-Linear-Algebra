/* C source code is found in dgemm_example.c */
// compiled with icc gemm.c -Wl,-rpath,/opt/intel/compilers_and_libraries_2017.4.181/mac/mkl/lib  -Wl,-rpath,/opt/intel/compilers_and_libraries_2017.4.181/mac/compiler/lib -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -qopenmp -mkl -liomp5

//source http://www.mscs.dal.ca/cluster/manuals/intel-mkl/examples/vmlc/source/
#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"


#define VEC_LEN 11



#define __SEXP_BEG -17.0
#define __SEXP_END 18.0



int main()
{
   float fA[VEC_LEN],fB1[VEC_LEN],fB2[VEC_LEN];

  int i=0,vec_len=VEC_LEN;
  double CurRMS,MaxRMS=0.0;

  for(i=0;i<vec_len;i++) {
    fA[i]=(float)(__SEXP_BEG+((__SEXP_END-__SEXP_BEG)*i)/vec_len);
    fB1[i]=0.0;
    fB2[i]=(float)exp(fA[i]);
  }

  vsExp(vec_len,fA,fB1);

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

  return 0;

}