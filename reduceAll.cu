#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

#define SMEMDIM 32
#define THREAD_PER_BLOCK 256

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float *idata,unsigned int n){
    for(int i=0;i<n;i++){
        idata[i] = 1.0f;
    }
}

float reduceOnCPU(float *idata,int size){
    int stride = 0;
    if(size == 1)return idata[0];

    stride = size/2;
    for(int i=0;i<stride;i++){
        idata[i] += idata[i+stride];
    }

    return reduceOnCPU(idata,stride);
}

//相邻规约,每个warp有分化
__global__ void reduce0(float *g_idata,float *g_odata,unsigned int n){
    __shared__ float smem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;

    if(idx<n)smem[tid] = g_idata[idx];
    else smem[tid] = 0;
    __syncthreads();

    for(int stride=1;stride<blockDim.x;stride *= 2){
        if(tid%(2*stride)==0){
            smem[tid] += smem[tid+stride];
        }
        __syncthreads();
    }
    if(tid == 0)g_odata[blockIdx.x] = smem[0];
}

//改变线程的索引，让同一个warp做相同的事情,最后一个warp还会有分化
__global__ void reduce1(float *g_idata,float *g_odata,unsigned int n){
    __shared__ float smem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;

    if(idx<n)smem[tid] = g_idata[idx];
    else smem[tid] = 0;
    __syncthreads();

    for(int stride=1;stride<blockDim.x;stride *= 2){
        int id = tid *2*stride;
        if(id<blockDim.x)
        smem[id] += smem[id+stride];
        __syncthreads();
    }
    if(tid == 0)g_odata[blockIdx.x] = smem[0];
}

//reduce1中0号线程读取0号和1号元素，16号线程读取32和33号元素，这就会产生bank conflict
//在这个核函数中把不同线程访问同一个bank的不同地址变为同一线程访问同一bank的不同地址。
//交错配对
__global__ void reduce2(float *g_idata,float *g_odata,unsigned int n){
    __shared__ float smem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;

    if(idx<n)smem[tid] = g_idata[idx];
    else smem[tid] = 0;
    __syncthreads();//一定要写同步，要不然不保证结果正确

    for(int stride=blockDim.x/2;stride>0;stride >>= 1){
        if(tid<stride)
        smem[tid] += smem[tid+stride];
        __syncthreads();
    }
    if(tid == 0)g_odata[blockIdx.x] = smem[0];
}

//线程展开，一个线程块处理两个数据块相加后的一个数据块，还是一个线程处理一个元素
__global__ void reduce3(float *g_idata,float *g_odata,unsigned int n){
    __shared__ float smem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*2+threadIdx.x;
    if((idx+blockDim.x)<n){
        g_idata[idx] += g_idata[idx+blockDim.x]; //全局内存上先做一次相加
        smem[tid] = g_idata[idx];
    }
    else smem[tid] = 0;
    __syncthreads();

    for(int stride=blockDim.x/2;stride>0;stride >>= 1){
        if(tid<stride){
            smem[tid] += smem[tid+stride];
        }
        __syncthreads();
    }
    if(tid == 0)g_odata[blockIdx.x] = smem[0];
}

//最后一个线程束都是执行同一条指令。没必要在进行一次同步
__global__ void reduce4(float *g_idata,float *g_odata,unsigned int n){
    __shared__ float smem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*2+threadIdx.x;
    if((idx+blockDim.x)<n){
        g_idata[idx] += g_idata[idx+blockDim.x]; //全局内存上先做一次相加
        smem[tid] = g_idata[idx];
    }
    else smem[tid] = 0;
    __syncthreads();

    for(int stride=blockDim.x/2;stride>32;stride >>= 1){
        if(tid<stride){
            smem[tid] += smem[tid+stride];
        }
        __syncthreads();
    }

    if(tid<32){
        volatile float *vmem = smem;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+ 8];
        vmem[tid] += vmem[tid+ 4];
        vmem[tid] += vmem[tid+ 2];
        vmem[tid] += vmem[tid+ 1];
    }
    if(tid==0)g_odata[blockIdx.x] = smem[0];
}

//块内的循环完全展开
__global__ void reduce5(float *g_idata,float *g_odata,unsigned int n){
    __shared__ float smem[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*2+threadIdx.x;
    if((idx+blockDim.x)<n){
        g_idata[idx] += g_idata[idx+blockDim.x]; //全局内存上先做一次相加
        smem[tid] = g_idata[idx];
    }
    else smem[tid] = 0;
    __syncthreads();

    if(blockDim.x>=512 && tid<256)smem[tid] += smem[tid+256];
    __syncthreads();//还是要写同步！
    if(blockDim.x>=256 && tid<128)smem[tid] += smem[tid+128];
    __syncthreads();
    if(blockDim.x>=128 && tid<64) smem[tid] += smem[tid+ 64];
    __syncthreads();

    if(tid<32){
        volatile float *vmem = smem;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+ 8];
        vmem[tid] += vmem[tid+ 4];
        vmem[tid] += vmem[tid+ 2];
        vmem[tid] += vmem[tid+ 1];
    }
    if(tid==0)g_odata[blockIdx.x] = smem[0];
}

//使用shuffle
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum){
    //0xffffffff表示所有线程都参与，都做+=
    if(blockSize >= 32)sum += __shfl_down_sync(0xffffffff,sum,16);
    if(blockSize >= 16)sum += __shfl_down_sync(0xffffffff,sum,8);
    if(blockSize >= 8)sum += __shfl_down_sync(0xffffffff,sum,4);
    if(blockSize >= 4)sum += __shfl_down_sync(0xffffffff,sum,2);
    if(blockSize >= 2)sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}
template <unsigned int blockSize>
__global__ void reduce6(float *g_idata,float *g_odata,unsigned int n){
    //每个块定义一个共享内存，块中最多有32个线程束，所以大小定义为32
    __shared__ float warpLevelSums[SMEMDIM];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x*blockIdx.x*2+threadIdx.x;

    float sum=0;
    //每个线程将这两个元素的sum存起来
    if(idx<n)sum += g_idata[idx]+g_idata[idx+blockDim.x];
    
    const int laneId = threadIdx.x%SMEMDIM;
    const int warpId = threadIdx.x/SMEMDIM;
    //归约每个线程束的结果到0号线程
    sum = warpReduceSum<blockSize>(sum);
    //放到共享内存中
    if(laneId == 0)warpLevelSums[warpId] = sum;
    __syncthreads();

    sum = (threadIdx.x <blockDim.x/SMEMDIM)?warpLevelSums[laneId]:0;

    if(warpId == 0)sum = warpReduceSum<blockSize/SMEMDIM>(sum);

    if(tid ==0 )g_odata[blockIdx.x] = sum;
}

//使用shuffle
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum){
    if(blockSize >= 32)sum += __shfl_down_sync(0xFFFFFFFF,sum,16);
    if(blockSize >= 16)sum += __shfl_down_sync(0xFFFFFFFF,sum,8);
    if(blockSize >= 8)sum += __shfl_dowm_sync(0xFFFFFFFF,sum,4);
    if(blockSize >= 4)sum += __shfl_down_sync(0xFFFFFFFF,sum,2);
    if(blockSize >= 2)sum += __shfl_down_sync(0xFFFFFFFF,sum,1);
    return sum;
}

template <unsigned int blockSize>
__global__ void reduce6(float *g_idata,float *g_odata,unsigned int n){
    float sum = 0.0f;
    unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    while(i<n){
        sum += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }

    __shared__ float warpLevelSum[warp_size];
    const int warpId = threadIdx.x / warp_size;
    const int laneId = threadIdx.x % warp_size;
    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0)warpLevelSum[warpId] = sum;
    __syncthreads();
    sum = (threadIdx.x<(blockDim.x/warp_size)) ? warpLevelSum[laneId] :0;
    if(warpId == 0)sum = warpReduceSum<blockSize/warp_size>(sum);
    if(tid == 0)g_odata[blockIdx.x] = sum;
}

int main(int argc,char **argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using dev:%d ,name:%s\n",dev,deviceProp.name);

    //定义规约长度，block和grid大小
    int size = 1<<25;
    int nBytes = size *sizeof(float);
    dim3 block(THREAD_PER_BLOCK);
    dim3 grid((size+block.x-1)/block.x);

    //定义主机内存指针
    float *h_idata,*h_odata,*tmp;
    h_idata = (float *)malloc(nBytes);
    tmp = (float *)malloc(nBytes);
    h_odata = (float *)malloc(grid.x*sizeof(float));
    memset(h_odata,0,grid.x*sizeof(float));

    //初始化数组
    initialData(h_idata,size);
    memcpy(tmp,h_idata,nBytes);

    //CPU进行规约
    double iStart,iElapes;
    iStart = cpuSecond();
    float cpu_sum = reduceOnCPU(tmp,size);
    iElapes = cpuSecond() - iStart;
    printf("reduce  on CPU result:%f,Using %f sec\n",cpu_sum,iElapes);

    //定义设备指针
    float *d_idata,*d_odata;
    float gpu_sum =0.0f;
    cudaMalloc((float **)&d_idata,nBytes);
    cudaMalloc((float **)&d_odata,grid.x*sizeof(float));

    //reduce0进行规约
    cudaMemcpy(d_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    reduce0<<<grid,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElapes = cpuSecond() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x;i++){
        gpu_sum += h_odata[i];
    }
    printf("reduce0 on GPU result:%f,Using %f sec\n",gpu_sum,iElapes);

    //reduce1进行规约
    gpu_sum = 0;
    cudaMemcpy(d_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    reduce1<<<grid,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElapes = cpuSecond() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x;i++){
        gpu_sum += h_odata[i];
    }
    printf("reduce1 on GPU result:%f,Using %f sec\n",gpu_sum,iElapes);

    //reduce2进行规约
    gpu_sum = 0;
    cudaMemcpy(d_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    reduce2<<<grid,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElapes = cpuSecond() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x;i++){
        gpu_sum += h_odata[i];
    }
    printf("reduce2 on GPU result:%f,Using %f sec\n",gpu_sum,iElapes);

    //reduce3进行规约,减少一半的块
    gpu_sum = 0;
    cudaMemcpy(d_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    reduce3<<<grid.x/2,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElapes = cpuSecond() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x/2*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x/2;i++){
        gpu_sum += h_odata[i];
    }
    printf("reduce3 on GPU result:%f,Using %f sec\n",gpu_sum,iElapes);

    //reduce4进行规约,减少一半的块并展开最后一个线程束
    gpu_sum = 0;
    cudaMemcpy(d_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    reduce4<<<grid.x/2,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElapes = cpuSecond() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x/2*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x/2;i++){
        gpu_sum += h_odata[i];
    }
    printf("reduce4 on GPU result:%f,Using %f sec\n",gpu_sum,iElapes);

    //reduce5进行规约,减少一半的块并展开最后一个线程束,并将块内的循环展开
    gpu_sum = 0;
    cudaMemcpy(d_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    reduce5<<<grid.x/2,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElapes = cpuSecond() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x/2*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x/2;i++){
        gpu_sum += h_odata[i];
    }
    printf("reduce5 on GPU result:%f,Using %f sec\n",gpu_sum,iElapes);

    //reduce6进行归约,使用洗牌指令，在线程束内进行数据交换。
    gpu_sum = 0;
    cudaMemcpy(d_idata,h_idata,nBytes,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    reduce6<THREAD_PER_BLOCK><<<grid.x/2,block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElapes = cpuSecond() - iStart;
    cudaMemcpy(h_odata,d_odata,grid.x/2*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<grid.x/2;i++){
        gpu_sum += h_odata[i];
    }
    printf("reduce6 on GPU result:%f,Using %f sec\n",gpu_sum,iElapes);


    // if(fabs(gpu_sum - cpu_sum)>0.00001)
    // printf("Match fail!\n");
    // else
    // printf("Match success!\n");

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    free(tmp);

    return 0;
}

// Using dev:0 ,name:Tesla T4
// reduce  on CPU result:33554432.000000,Using 0.107859 sec
// reduce0 on GPU result:33554432.000000,Using 0.005111 sec
// reduce1 on GPU result:33554432.000000,Using 0.004243 sec
// reduce2 on GPU result:33554432.000000,Using 0.003343 sec
// reduce3 on GPU result:33554432.000000,Using 0.001843 sec
// reduce4 on GPU result:33554432.000000,Using 0.001222 sec
// reduce5 on GPU result:33554432.000000,Using 0.001291 sec
// reduce6 on GPU result:33554432.000000,Using 0.000926 sec