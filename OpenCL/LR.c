

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define BUFF 6536

char buffer [BUFF];
//gcc clexp.c  -framework OpenCL
//gcc clexp.c -O3 -framework OpenCL
////////////////////////////////////////////////////////////////////////////////



const char *KernelSource3 = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* input2,                                             \n"\
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * log(input2[i]) + (( 1 - input[i]) *  log(1 - input2[i]));\n" \
"}                                                                      \n" \
"\n";
const char *KernelSource4 = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* input2,                                             \n"\
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] - input2[i];                        ;\n" \
"}                                                                      \n" \
"\n";


const char *KernelSource6 = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* input2,                                             \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count, const int M, const int N, const float b)  \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   int j = get_global_id(1);                                                                   \n"\
"   float sum = 0.0f;                                                    \n" \
"   int a = N * i;                                                          \n"\
"   int c = j * 64;                                                          \n"\
"   __local float sums[256];                                              \n"\
"   //__attribute__((opencl_unroll_hint))                                 \n"\

"   for(int k = 0; k < 64; k++){                                      \n" \
"       int s = k+c;                                                 \n"\
"       sum += input[a + s ] * input2[s];                                  \n" \

"   }                                                                     \n" \
" sums[j] = sum;                                                       \n"\
"  barrier(CLK_LOCAL_MEM_FENCE);                                        \n"\
"  if (j == 0)                                                      \n"\
"  {                                                                \n"\
"   float sumtotal = 0.0f;                                             \n"\

"   for (int p = 0; p < 256 ; p++)                                            \n"\
"   {                                                                       \n"\
"       sumtotal += sums[p];                                               \n"\
"   }                                                                      \n"\
"  output[i] = sumtotal -b;                                                           \n"\
"  }                                                                          \n" \

"}                                                                          \n" \

"\n";

const char *KernelSource7 = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* input2,                                                \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count, const int M, const int N, const float NR)                                            \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   //int f = (i/2)+1;                                                    \n" \
"   float acc = 0.0;                                               \n" \
"   for(int j = 0; j < M; j++){                                           \n" \
"       acc += input[N * j + i] * input2[j];                        \n" \
"   }                                                               \n" \
"   output[i] = acc/NR;                                                \n" \
"}                                                                      \n" \
"\n";


////////////////////////////////////////////////////////////////////////////////

struct clDataOperation {
    cl_device_type   device_type;
    cl_device_id     device_id;
    cl_context       context;
    cl_command_queue commands;
    cl_program       program;
    cl_kernel        kernel;
    cl_mem input;                       // device memory used for the input array
    cl_mem input2;
    cl_mem output;
    
};
int sigmoid(float * data, float * results, int NR, int DATA_SIZE);
int add(float * data, float b, float * results, int DATA_SIZE);
void propagate(float* w, float b, float* x, float* y, float* dw,  float* db, int N, int M, int NR, struct clDataOperation *kernelData, struct clDataOperation *kernelData2, int numloop);
int costVec(float * data, float * data2, float * results, int DATA_SIZE);
void subtract(float* y, float* A, float* result, int N, int NR);
void calcGradients(float* x, float* E, float* y, float* dw, float* db, int N, int M,  int NR, struct clDataOperation *kernelData);
void subtract(float* data, float* data2, float* result, int DATA_SIZE, int NR);
void scalarDiv(float * data, float b, float * results, int DATA_SIZE);
int dotmv(float * data, float * data2, int M, int N, float * results, int DATA_SIZE, float b, int NR, struct clDataOperation *kernelData);

int dotmvt(float * data, float * data2, int M, int N, float * results, int DATA_SIZE, float NR, struct clDataOperation *kernelData);


void optimize(float* w, float* b, float* x, float* y, float* dw,  float* db, int N, int M, int MR, int NR, int epoch, float lRate, struct clDataOperation *kernelData, struct clDataOperation *kernelData2);
int update(float* w, float* dw, float lRate, int N, float* results);
void predict(float* w, float b, float* x, float* y, float* dw,  float* db, int N, int M, int NR, struct clDataOperation *kernelData);
void startKernel( struct clDataOperation *kernelData, float * data, int M, int N , int NR);
void startKernel2( struct clDataOperation *kernelData, float * data, int M, int N, float NR );
void finishedWithKernelData( struct clDataOperation *kernelData);



void printMAT(float *MAT,int N, int M){
    
    printf("N = %d \n", N);
    for (int i = 0; i <  N; i = i + 500){
        
        printf(" %f ",*(MAT + i + 32000));
        
        
    }
}
double get_time() //https://stackoverflow.com/questions/2349776/how-can-i-benchmark-c-code-easily
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}



int main(int argc, char** argv){
    
    unsigned int M = 256;           //height  vector legnth
    unsigned int N = 16384;          //width
    unsigned int count = 256;
    int NR = 209;
    int MR = 12288;
    float *X = (float *)calloc(N * M, sizeof(float));
    float *w = (float *)calloc(N , sizeof(float));
    float *dw = (float *)calloc(N , sizeof(float));
    
    float* b = (float *)calloc(1, sizeof(float));
    float* db = (float *)calloc(1, sizeof(float));
    int i = 0;
    int j = 0;
    FILE *fp = fopen("data/dataset.csv", "r");
    if(fp != NULL){
        while(fgets(buffer, 2000, (FILE*) fp)) {
            
            for (char *p = strtok(buffer,","); p != NULL; p = strtok(NULL, ","))
            {
                float f = (float)atof(p);
                *(X + i*N + j) = f/255;
                i++;
            }
            i = 0;
            j++;
        }
        fclose(fp);
    }
    float *Y = (float *)calloc(M, sizeof(float));
    i = 0;
    j = 0;
    FILE *fpy = fopen("data/datasety.csv", "r");
    if(fpy != NULL){
        while(fgets(buffer, 800, (FILE*) fpy)) {
            
            for (char *p = strtok(buffer,","); p != NULL; p = strtok(NULL, ","))
            {
                float f = (float)atof(p);
                *(Y + i) = f;
                i++;
            }
        }
        fclose(fpy);
    }
    int epoch = 2000;
    float lRate = 0.005;
    
    
    
    struct clDataOperation kernelData;
    startKernel(&kernelData, X, M, N, NR);
    struct clDataOperation kernelData2;
    startKernel2(&kernelData2, X, M, N, (float)NR);
    
    
    double start =get_time();
    
    
    //float start =get_time();
    
    
    
    optimize(w,b,X, Y, dw, db,  N,  M, MR, NR, epoch, lRate, &kernelData, &kernelData2);
    double bench = (get_time() - start);
    printf("Time = %f \n", bench);
    
   
    
    predict(w,*(b),X, Y, dw, db,  N,  M, NR,  &kernelData);
    
    
    
    
    M = 256;           //height  vector legnth
    N = 16384;          //width
    count = 256;
    NR = 50;
    MR = 12288;
    float *Xtest = (float *)calloc(N * M, sizeof(float));
    i = 0;
    j = 0;
    FILE *fpt = fopen("data/datasettest.csv", "r");
    if(fpt != NULL){
        while(fgets(buffer, 2000, (FILE*) fpt)) {
            
            for (char *p = strtok(buffer,","); p != NULL; p = strtok(NULL, ","))
            {
                float f = (float)atof(p);
                *(Xtest + i*N + j) = f/255;
                i++;
                //printf("xxxxhere");
            }
            i = 0;
            j++;
        }
        fclose(fpt);
    }
    float *Ytest = (float *)calloc(M, sizeof(float));
    i = 0;
    j = 0;
    FILE *fpyt = fopen("data/datasettesty.csv", "r");
    if(fpyt != NULL){
        while(fgets(buffer, 800, (FILE*) fpyt)) {
            
            for (char *p = strtok(buffer,","); p != NULL; p = strtok(NULL, ","))
            {
                float f = (float)atof(p);
                *(Ytest + i) = f;
                // printf("here%f", *(Ytest + i));
                i++;
            }
        }
        fclose(fpyt);
    }
    
    struct clDataOperation kernelData3;
    startKernel(&kernelData3, Xtest, M, N, NR);
    
    //predict(w,*(b),X, Y, dw, db,  N,  M, NR,  *kernelData3);
    
    predict(w,*(b),Xtest, Ytest, dw, db,  N,  M, NR, &kernelData3);
    finishedWithKernelData(&kernelData);
    finishedWithKernelData(&kernelData2);
    finishedWithKernelData(&kernelData3);
    
    
}
void finishedWithKernelData( struct clDataOperation *kernelData){
    clReleaseMemObject(kernelData->input);
    clReleaseMemObject(kernelData->input2);
    clReleaseMemObject(kernelData->output);
    clReleaseProgram(kernelData->program);
    clReleaseKernel(kernelData->kernel);
    clReleaseCommandQueue(kernelData->commands);
    clReleaseContext(kernelData->context);

    
    
}
void startKernel( struct clDataOperation *kernelData, float * data, int M, int N, int NR ){
    int err;
    cl_uint deviceCount;
    cl_device_id* devices;
    
    
    
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
    
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, 0, NULL, &valueSize);
    //value = (char*) malloc(valueSize);
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, valueSize, value, NULL);
    //printf("%d. Device: %s\n", 2+1, value);
    kernelData->device_id = devices[2];
    
    
    kernelData->context = clCreateContext(NULL, 1, &kernelData->device_id, NULL, NULL, NULL);
    kernelData->commands = clCreateCommandQueue(kernelData->context,kernelData->device_id, 0, NULL);
    kernelData->program = clCreateProgramWithSource(kernelData->context, 1, (const char **) & KernelSource6, NULL, &err);
    err = clBuildProgram(kernelData->program, 0, NULL, NULL, NULL, NULL);
    kernelData->kernel = clCreateKernel(kernelData->program, "square", &err);
    
    kernelData->input = clCreateBuffer(kernelData->context,  CL_MEM_READ_ONLY,  sizeof(float) * M * N, NULL, NULL);
    kernelData->input2 = clCreateBuffer(kernelData->context,  CL_MEM_READ_ONLY,  sizeof(float) * N, NULL, NULL);
    kernelData->output = clCreateBuffer(kernelData->context, CL_MEM_WRITE_ONLY, sizeof(float) * M, NULL, NULL);
    
    
    err = clEnqueueWriteBuffer(kernelData->commands, kernelData->input, CL_TRUE, 0, sizeof(float) *  M * N, data, 0, NULL, NULL);
    
    
    err  = clSetKernelArg(kernelData->kernel, 0, sizeof(cl_mem), &kernelData->input);
    err  = clSetKernelArg(kernelData->kernel, 1, sizeof(cl_mem), &kernelData->input2);
    err |= clSetKernelArg(kernelData->kernel, 2, sizeof(cl_mem), &kernelData->output);
    err |= clSetKernelArg(kernelData->kernel, 3, sizeof(unsigned int), &NR);
    err |= clSetKernelArg(kernelData->kernel, 4, sizeof(int), &M);
    err |= clSetKernelArg(kernelData->kernel, 5, sizeof(int), &N);
    
}

void startKernel2( struct clDataOperation *kernelData, float * data, int M, int N, float NR){
    
    int err;
    
    cl_uint deviceCount;
    cl_device_id* devices;
    
    unsigned int count = N;
    char* value;
    size_t valueSize;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
    
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, 0, NULL, &valueSize);
    //value = (char*) malloc(valueSize);
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, valueSize, value, NULL);
    //printf("%d. Device: %s\n", 2+1, value);
    kernelData->device_id = devices[2];
    
    
    kernelData->context = clCreateContext(NULL, 1, &kernelData->device_id, NULL, NULL, NULL);
    kernelData->commands = clCreateCommandQueue(kernelData->context, kernelData->device_id, 0, NULL);
    cl_event event = NULL;
    
    kernelData->program = clCreateProgramWithSource(kernelData->context, 1, (const char **) & KernelSource7, NULL, &err);
    
    err = clBuildProgram(kernelData->program, 0, NULL, NULL, NULL, NULL);
    
    kernelData->kernel = clCreateKernel(kernelData->program, "square", &err);
    
    kernelData->input = clCreateBuffer(kernelData->context,  CL_MEM_READ_ONLY,  sizeof(float) * M * N, NULL, NULL);
    kernelData->input2 = clCreateBuffer(kernelData->context,  CL_MEM_READ_ONLY,  sizeof(float) * N, NULL, NULL);
    kernelData->output = clCreateBuffer(kernelData->context, CL_MEM_WRITE_ONLY, sizeof(float) * N, NULL, NULL);
    err = clEnqueueWriteBuffer(kernelData->commands, kernelData->input, CL_TRUE, 0, sizeof(float) *  M * N, data, 0, NULL, NULL);
    
    err = 0;
    err  = clSetKernelArg(kernelData->kernel, 0, sizeof(cl_mem), &kernelData->input);
    err  = clSetKernelArg(kernelData->kernel, 1, sizeof(cl_mem), &kernelData->input2);
    err |= clSetKernelArg(kernelData->kernel, 2, sizeof(cl_mem), &kernelData->output);
    err |= clSetKernelArg(kernelData->kernel, 3, sizeof(unsigned int), &count);
    err |= clSetKernelArg(kernelData->kernel, 4, sizeof(int), &M);
    err |= clSetKernelArg(kernelData->kernel, 5, sizeof(int), &N);
    err |= clSetKernelArg(kernelData->kernel, 6, sizeof(float), &NR);
}


void optimize(float* w, float* b, float* x, float* y, float* dw,  float* db, int N, int M, int MR, int NR, int epoch, float lRate, struct clDataOperation *kernelData, struct clDataOperation *kernelData2){
    
    for(int e = 0;e < epoch; e++){
        propagate(w,*(b),x, y, dw, db,  N,  M, NR, kernelData, kernelData2, e);
        
        for(int i = 0; i < N; i++){
            *(w + i) = *(w + i) - lRate *(*(dw + i));
            
        }
        *b = *b - lRate* *(db);
        
    }
}


void predict(float* w, float b, float* x, float* y, float* dw,  float* db, int N, int M, int NR, struct clDataOperation *kernelData){
    float *C = (float *)calloc(M, sizeof(float));
    
    float *D = (float *)calloc(N, sizeof(float));
    float *E = (float *)calloc(N, sizeof(float));
    float *F = (float *)calloc(N, sizeof(float));
    dotmv(x, w, M, N, C, M, b, NR, kernelData);
    for(int i = 0; i < NR; i++){
        *(E + i) = 1.00 / (1.0 + exp(-*(C + i)));
        
    }
    
    
    for (int i = 0; i <  NR; i ++){
        //printf("prob %f\n", *(E + i));
        if(*(E + i) > 0.5){
            *(E + i) = 1;
        }else{
            *(E + i) = 0;
        }
        //printf("Predicted = %f \n Actual = %f\n", *(E + i), *(y +i ));
    }
    int same = 0;
    int diff = 0;
    for (int i = 0; i <  NR; i ++){
        //printf("Predicted = %f \n Actual = %f\n", *(E + i), *(y +i ));
        if(*(E + i) ==*(y +i )){
            same++;
        }else{
            diff++;
        }
        
    }
    printf("\n\nsame %d, diff %d \n\n", same, diff);
    
    
}



void propagate(float* w, float b, float* x, float* y, float* dw,  float* db, int N, int M, int NR, struct clDataOperation *kernelData, struct clDataOperation *kernelData2, int numLoop){
    float *C = (float *)calloc(M, sizeof(float));
    
    float *D = (float *)calloc(N, sizeof(float));
    //float *E = (float *)calloc(N, sizeof(float));
    float *F = (float *)calloc(N, sizeof(float));
    
    dotmv(x, w, M, N, C, M, b, NR, kernelData);
    
    //printf("Predicted = %f ", *(C));
    //sigmoid
    
    
    
    
    for(int i = 0; i < NR; i++){
        *(D + i) = 1.00 / (1.0 + exp(-*(C + i)));
        
    }
    
    /*
    float acc = 0;
    for (int i = 0; i <  M; i ++){
        
        acc = 0;
        float t = 0;
        for(int j = 0; j < N; j++){
            t = *(x+N * i + j) * *(w + j);
            acc += t;
            //if(t != 0.0 )
            //printf(" accume += %f * %f = %f so accume = %f",*(x+N * i + j),*(w + j), t, acc );
        }
        
        acc -= b;
        *(C+i) = 1.00 / (1.0 + exp(-acc));
    }
    */

    //add(C, b, D, M);
    //sigmoid(C, E, NR,  M);
    
    if(numLoop % 100 == 0){
        costVec(y,D,F, M);
        float accume = 0;
        for(int j = 0; j < NR; j++){
            accume = accume + *(F + j);
        }
        float cost = accume/(-NR);
        printf("cost: %f\n",cost);
    }
    /*
    */
    
    calcGradients(x, D, y, dw,db,  N, M, NR, kernelData2);
    
    
}

void calcGradients(float* x, float* E, float* y, float* dw, float* db, int N, int M, int NR, struct clDataOperation *kernelData){
    float *G = (float *)calloc(N, sizeof(float));
    float *H = (float *)calloc(N, sizeof(float));
    //subtract(E, y, G, M, NR);
     float accume = 0;
    for(int j = 0; j < NR; j++){
         *(G + j) = *(E + j) - *(y + j);
        accume = accume + *(G + j);

    }
    *db = accume/(NR);
    
   
    
    
    dotmvt(x, G, M, N, dw,N, NR, kernelData);
    /*
    for (int i = 0; i <  N; i ++){
        
        float acc = 0;
        float t = 0;
        for(int j = 0; j < M; j++){
            t = *(x+N * j + i) * *(G + j);
            acc += t;
            //if(t != 0.0 )
            //printf(" accume += %f * %f = %f so accume = %f",*(x+N * i + j),*(w + j), t, acc );
        }
        *(dw+i) = acc/NR;
    }
    
    
    */
    

    
}

int dotmv(float * data, float * data2, int M, int N, float * results, int DATA_SIZE, float b, int NR, struct clDataOperation *kernelData){
    int err;
    int i = 0;// error code returned from api calls
    
    // results returned from device
    unsigned int correct;               // number of correct results returned
    
    //size_t global[2];                      // global domain size for our calculation
    //size_t local[2];                       // local domain size for our calculation
    
    unsigned int count = DATA_SIZE;
    char* value;
    size_t valueSize;
 
    err = clEnqueueWriteBuffer(kernelData->commands, kernelData->input2, CL_TRUE, 0, sizeof(float) *  N, data2, 0, NULL, NULL);
    
    err = 0;
    
    
    err |= clSetKernelArg(kernelData->kernel, 6, sizeof(float), &b);
    
    
    unsigned int count2 = 20;
    //global[0] = count;
   // global[1] = count2;
    
    //local[0] = count2;
    //local[1] = count2;
    //err = clGetKernelWorkGroupInfo(kernelData->kernel, kernelData->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    size_t global[2] = {count, 256 };
    size_t  local[2] = {1, 256};
    
    err = clEnqueueNDRangeKernel(kernelData->commands, kernelData->kernel, 2, NULL, global, local, 0, NULL, NULL);
    
    clFinish(kernelData->commands);
    err = clEnqueueReadBuffer( kernelData->commands, kernelData->output, CL_TRUE, 0, sizeof(float) * M, results, 0, NULL, NULL );
    return 0;
}

int dotmvt(float * data, float * data2, int M, int N, float * results, int DATA_SIZE, float NR, struct clDataOperation *kernelData){
    int err;
    unsigned int correct;               // number of correct results returned
        
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    unsigned int count = N;
    cl_uint deviceCount;
    
    clEnqueueWriteBuffer(kernelData->commands, kernelData->input2, CL_TRUE, 0, sizeof(float) *  N, data2, 0, NULL, NULL);
    
    clGetKernelWorkGroupInfo(kernelData->kernel, kernelData->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    global = count;
    //size_t  local[1] = {256};
    clEnqueueNDRangeKernel(kernelData->commands, kernelData->kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(kernelData->commands);
    clEnqueueReadBuffer( kernelData->commands, kernelData->output, CL_TRUE, 0, sizeof(float) * N, results, 0, NULL, NULL );
    
        
    return 0;
}
int costVec(float * data, float * data2, float * results, int DATA_SIZE){
    
    int err;
    int i = 0;// error code returned from api calls
    
    // results returned from device
    unsigned int correct;               // number of correct results returned
    
    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
    
    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem input2;
    cl_mem output;                      // device memory used for the output array
    
    
    cl_uint deviceCount;
    cl_device_id* devices;
    
    
    
    unsigned int count = DATA_SIZE;
    char* value;
    size_t valueSize;
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
    devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
    
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, 0, NULL, &valueSize);
    //value = (char*) malloc(valueSize);
    //clGetDeviceInfo(devices[2], CL_DEVICE_NAME, valueSize, value, NULL);
    //printf("%d. Device: %s\n", 2+1, value);
    device_id = devices[2];
    
    
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    commands = clCreateCommandQueue(context, device_id, 0, NULL);
    cl_event event = NULL;
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource3, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "square", &err);
    
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    input2 = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    
    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(float) * count, data2, 0, NULL, NULL);
    
    err = 0;
    
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
    
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    
    global = count;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    
    clFinish(commands);
    
    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );
    clReleaseMemObject(input);
    clReleaseMemObject(input2);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    
    return 0;
    
}
