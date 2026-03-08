#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float *a,unsigned int nx,unsigned int ny){
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            a[i*nx+j] = 1.0f;
        }
    }
}

void transposeOnHost(float *a,float *b,unsigned int nx,unsigned int ny){
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++){
            b[j*ny+i] = a[i*nx+j];
        }
    }
}

__global__ void copyrow(float *a,float *b,unsigned int nx,unsigned int ny){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int idy = blockDim.y*blockIdx.y+threadIdx.y;
    if(idx<nx && idy<ny){
        b[idx*ny+idy] = a[idx*ny+idy];
    }
}

__global__ void copycol(float *a,float *b,unsigned int nx,unsigned int ny){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int idy = blockDim.y*blockIdx.y+threadIdx.y;
    if(idx<nx && idy<ny){
        b[idy*nx+idx] = a[idy*nx+idx];
    }
}

__global__ void transposeOnDevice(float *a,float *b,unsigned int nx,unsigned int ny){
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    int idy = blockDim.y*blockIdx.y+threadIdx.y;
    if(idx<nx && idy<ny){
        b[idx*ny+idy] = a[idy*nx+idx];
    }
}

__global__ void transposeSmem(float *in,float * out,unsigned int nx,unsigned int ny){
    __shared__ float tile[BIDMY][BIDMX];
    unsigned int ix,iy,ti,to;
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    ti = iy * nx + ix;

    unsigned int bid,irow,icol;
    bid = threadIdx.y * blockDim.x +threadIdx.x;
    irow = bid / blockDim.y;
    icol = bid % blockDim.y;

    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;
    to = iy * ny + ix;

    if(ix<nx && iy<ny){
        tile[threadIdx.y][threadIdx.x] = in[ti];
        __syncthreads();
        out[to] = tile[icol][irow];
    }
}

int checkResult(float *h_a,float *d_a,)

int main(int argc,char **argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using device %d: %s\n",dev,deviceProp.name);
    cudaSetDevice(dev);

    int nx = 1<<11;
    int ny = 1<<11;
    int nSize = nx*ny*sizeof(float);

    int blockx = 16;
    int blocky = 16;

    dim3 block(blockx,blocky);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);

    float *h_a,*h_b;
    h_a = (float *)malloc(nSize);
    h_b = (float *)malloc(nSize);

    initialData(h_a,nx,ny);
    transposeOnHost(h_a,h_b,nx,ny);

    double iStart,iElapes;
    float *d_a,*d_b;
    cudaMalloc((float **)&d_a,nSize);
    cudaMalloc((float **)&d_b,nSize);

    cudaMemcpy(d_a,h_a,nSize,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    copyrow<<<grid,block>>>(d_a,d_b,nx,ny);
    cudaDeviceSynchronize();
    iElapes = cpuSecond()-iStart;
    printf("copyrow using %f sec\n",iElapes);

    cudaMemcpy(d_a,h_a,nSize,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    copycol<<<grid,block>>>(d_a,d_b,nx,ny);
    cudaDeviceSynchronize();
    iElapes = cpuSecond()-iStart;
    printf("copycol using %f sec\n",iElapes);

    cudaMemcpy(d_a,h_a,nSize,cudaMemcpyHostToDevice);
    iStart = cpuSecond();
    transposeOnDevice<<<grid,block>>>(d_a,d_b,nx,ny);
    cudaDeviceSynchronize();
    iElapes = cpuSecond()-iStart;
    printf("transposeOnDevice using %f sec\n",iElapes);

    cudaFree(d_a);
    cudaFree(d_b);

    free(h_a);
    free(h_b);
    cudaDeviceReset();

    return 0;
}