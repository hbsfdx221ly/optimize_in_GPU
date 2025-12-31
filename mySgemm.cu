
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

//cuda runtime
#include <cuda_runtime.h>

//宏定义
#define OFFSET(row,col,ld) ((row)*(ld) + (col))
//transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

//封装check cudaError
#define CHECK(func){                \
    cudaError_t error = (func);     \
    if(error != cudaSuccess){       \
        printf("error :%s ,%d ,%s \n",__FILE__,__LINE__,cudaGetErrorString(error));\
    }\
}

//模板
template <
    const int BLOCK_SIZE_M,             //加载到shared memory里A矩阵的行数
    const int BLOCK_SIZE_K,             //加载到shared memory里A矩阵的列数和B矩阵的行数
    const int BLOCK_SIZE_N,             //加载到shared memory里B矩阵的列数
    const int THREAD_SIZE_X,            //加载到线程register里一小行
    const int THREAD_SIZE_Y,            //加载到线程register里一小列
    const bool ENABLE_DOUBLE_BUFFER     //是否开启双缓存
    >
__global__ void sgemm(
    float *__restrict__ A,      //A,B,C分别接受主函数的A,B,C矩阵
    float *__restrict__ B,
    float *__restrict__ C,
    const int M,                //矩阵大小
    const int K,
    const int N
){
    //block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //thread num in x,y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;

    //used threads
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    //thread index in block
    int tid = ty * THREAD_X_PER_BLOCK + tx;

    //shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M]; //转置后A矩阵的大小
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N]; 

    //register for threads
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};    //存放计算结果
    float frag_a[2][THREAD_SIZE_Y];             //存放从shared mem里读取的一小列
    float frag_b[2][THREAD_SIZE_X];             //存放从shared mem里读取的一小行

    //定义临时寄存器，存放从global mem读取的数据
    //一次按行取4个float,要搬运ldg_a_num次
    const int ldg_a_num = BLOCK_SIZE_M * BLOCK_SIZE_K /(THREAD_NUM_PER_BLOCK*4);
    const int ldg_b_num = BLOCK_SIZE_K * BLOCK_SIZE_N /(THREAD_NUM_PER_BLOCK*4);
    float ldg_a_reg[4*ldg_a_num];
    float ldg_b_reg[4*ldg_b_num];

    //定义每行的线程数，线程开始读取时的行，列，步长
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW*4;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW*4;

    const int A_TILE_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    //一次读取A矩阵的BLOCK_SIZE_K列和B矩阵的BLOCK_SIZE_K行
    A = &A[(BLOCK_SIZE_M * by)*K];
    B = &B[BLOCK_SIZE_N * bx];

    //开始预取数据
    //从global mem里读取数据,对于A块先加载到临时寄存器当中,在转置加载到shared mem
    #pragma unroll
    for(int i = 0;i<BLOCK_SIZE_M;i+=A_TILE_STRIDE){
        int ldg_index = i / A_TILE_STRIDE*4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            A_TILE_ROW_START + i,
            A_TILE_COL,
            K)]);
        
        As[0][A_TILE_COL][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+3];
    }
    //对于B块从global mem加载数据到shared mem
    #pragma unroll
    for(int i = 0;i<BLOCK_SIZE_K;i+=B_TILE_STRIDE){
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START+i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START+i,
            B_TILE_COL,
            N)]);
    }
    __syncthreads();

    //现在数据已经加载到了shared mem里了，现在预取数据到线程的register。直接从shared mem里取数据不需要用偏移
    //从As里预取一小列数据
    #pragma unroll
    for(int thread_y = 0;thread_y<THREAD_SIZE_Y;thread_y += 4){
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    //从Bs里预取数据
    #pragma unroll
    for(int thread_x = 0;thread_x<THREAD_SIZE_X;thread_x += 4){
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    //数据初始化预取完成，现在开始循环预取数据和计算数据的过程
    int tile_idx = 0;           //用于从global mem读取数据的大循环
    int write_stage_idx = 1;    //用于指定对双缓存的哪一部分进行操作

    do{
        tile_idx += BLOCK_SIZE_K;
        if(tile_idx < K){
            #pragma unroll
            //从global mem里取数据到临时寄存器,先不放到shared mem里
            for(int i=0;i<BLOCK_SIZE_M;i += A_TILE_STRIDE){
                int ldg_index = i / A_TILE_STRIDE*4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    A_TILE_ROW_START+i,
                    A_TILE_COL+tile_idx,
                    K)]);
            }
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_K;i+=B_TILE_STRIDE){
                int ldg_index = i / B_TILE_STRIDE*4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    B_TILE_ROW_START+tile_idx+i,
                    B_TILE_COL,
                    N)]);
            }
        }

         int load_stage_idx = write_stage_idx ^1;
            //再从上一轮的shared mem里取数据到register,并进行计算,进行BLOCK_SIZE_K-1轮循环
        #pragma unroll
        for(int j=0;j<BLOCK_SIZE_K-1;++j){
            #pragma unroll
            for(int thread_y=0;thread_y<THREAD_SIZE_Y;thread_y += 4){
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][
                    THREAD_SIZE_Y * ty + thread_y 
                ]);
            }
            #pragma unroll
            for(int thread_x=0;thread_x<THREAD_SIZE_X;thread_x += 4){
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][
                    THREAD_SIZE_X * tx + thread_x
                ]);
            }

            //计算数据并临时存储
            #pragma unroll
            for(int thread_y=0;thread_y<THREAD_SIZE_Y;++thread_y){
                #pragma unroll
                for(int thread_x=0;thread_x<THREAD_SIZE_X;++thread_x){
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        //此时将临时寄存器上的数据加载到shared mem,并计算上一轮的最后一次小迭代
        if(tile_idx < K){
            //加载到As里
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_M;i+=A_TILE_STRIDE){
                int ldg_index = i / A_TILE_STRIDE*4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+3];
            }

            //加载到Bs里
            #pragma unroll
            for(int i=0;i<BLOCK_SIZE_K;i+=B_TILE_STRIDE){
                int ldg_index = i / B_TILE_STRIDE*4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START+i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }

            //加载到shared mem 需要同步
            __syncthreads();
            write_stage_idx ^= 1;
        }
        
        //第一个数据块的前BLOCK_SIZE_K-1次小迭代已完成，一部分寄存器空，一部分寄存器存着最后一次小迭代的数据
        //从第二个共享内存加载下一次数据到register
        #pragma unroll
        for(int thread_y=0;thread_y<THREAD_SIZE_Y;thread_y += 4){
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][
                THREAD_SIZE_Y * ty + thread_y
            ]);
        }
        #pragma unroll
        for(int thread_x=0;thread_x<THREAD_SIZE_X;thread_x += 4){
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][
                THREAD_SIZE_X * tx + thread_x
            ]);
        }
        //计算小迭代
        #pragma unroll
        for(int thread_y=0;thread_y<THREAD_SIZE_Y;++thread_y){
            #pragma unroll
            for(int thread_x=0;thread_x<THREAD_SIZE_X;++thread_x){
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx < K);

    //所有block都计算完成，最后将数据写回global mem
    #pragma unroll
    for(int thread_y=0;thread_y<THREAD_SIZE_Y;++thread_y){
        #pragma unroll
        for(int thread_x=0;thread_x<THREAD_SIZE_X;thread_x += 4){
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + THREAD_SIZE_Y * ty + thread_y,
                BLOCK_SIZE_N * bx + THREAD_SIZE_X * tx + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

int main(int argc,char **argv){
    if(argc != 4){
        printf("Using... ./main [M] [K] [N]\n");
        exit(0);
    }

    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    assert( M%8==0);
    assert( K%8==0);
    assert( N%8==0);

    size_t nbytes_A = M*K*sizeof(float);
    size_t nbytes_B = K*N*sizeof(float);
    size_t nbytes_C = M*N*sizeof(float);

    float *h_A,*h_B,*h_C,;
    h_A = (float *)malloc(nbytes_A);
    h_B = (float *)malloc(nbytes_B);
    h_C = (float *)malloc(nbytes_C);

    float *d_A,*d_B,*d_C;
    CHECK(cudaMalloc(&d_A,nbytes_A));
    CHECK(cudaMalloc(&d_B,nbytes_B));
    CHECK(cudaMalloc(&d_C,nbytes_C));

    //初始化A矩阵
    for(int i=0;i<M*K;i++){
        h_A[i] = i / 13;
    }

    //初始化B矩阵
    for(int i=0;i<K*N;i++){
        h_B[i] = i % 13;
    }

    CHECK(cudaMemcpy(d_A,h_A,nbytes_A,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B,h_B,nbytes_B,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C,h_C,nbytes_C,cudaMemcpyHostToDevice));

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    float msecTotal = 0;
    double msecMul[2] = {0};
    double gigaFlops[2] = {0};
    double flops = 2 * M * N * K;
    int nItr = 1000;

    cudaEvent_t start,stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    //多次计算求平均值
    for(int i=0;i<nItr;i++){
        dim3 block(BLOCK_SIZE_N / THREAD_SIZE_X,BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 grid(N / BLOCK_SIZE_N,M / BLOCK_SIZE_M);
        sgemm<BLOCK_SIZE_M,BLOCK_SIZE_K,BLOCK_SIZE_N,THREAD_SIZE_X,THREAD_SIZE_Y,ENABLE_DOUBLE_BUFFER><<<grid,block>>>(
            d_A,d_B,d_C,M,K,N
        );
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&msecTotal,start,stop));

    msecMul[0] = msecTotal / 1000;
    gigaFlops[0] = (flops * 1.0e-9) / (msecMul[0] / 1000.f);
    printf("mySgemm load %f ,gigaFlops : %f ,msecMul : %f",flops,gigaFlops[0],msecMul[0]);

    //没有写与cublas的性能对比
    
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

}
    