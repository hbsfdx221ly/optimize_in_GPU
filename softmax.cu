#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(float *input,float *output,const int M,const int N){
    const int tid = threadIdx.x;
    const int warpId = tid / warpSize;
    const int laneId = tid % warpSize;
    const int warpPerBlock = blockDim.x / warpSize;
    const int warpnum = warpPerBlock * gridDim.x;
    const int idx = warpPerBlock * blockDim.x + warpId;

    for(int m = idx;m<M;m+=warpnum){
        float maxval = -INFINITY;
        float sumval = 0.0f;
        float *x = input + m*N;
        float *y = output + m*N;
        for(int i = laneId;i<N;i+=warpSize){
            float bigger = fmaxf(maxval,x[i]);
            sumval = sumval * expf(maxval - bigger) + expf(x[i] - bigger);
            maxval = bigger;
        }
        float offsetmax,offsetsum;
        for(int offset = warpSize/2;offset > 0;offset >>= 1){
            offsetmax = __shfl_xor_sync(0xFFFFFFFF,maxval,offset);
            offsetsum = __shfl_xor_sync(0xFFFFFFFF,sumval,offset);
            if(offsetmax > maxval){
                sumval *= expf(maxval - offsetmax);
                maxval = offsetmax;
            }else{
                offsetsum *= expf(offsetmax - maxval);
            }
            sumval += offsetsum;
        }
        for(int i = laneId;i<N;i+=warpSize){
            y[i] = expf(x[i] - maxval) / sumval;
        }
    }
}