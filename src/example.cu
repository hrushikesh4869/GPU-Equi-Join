#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <thread>

using namespace std;

__global__ void write_output(int* input, int *output, int* buffer1, int* index, int* current_buffer, int* count, int *isfull, int bufferSize, int dataSize){
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("inside the kernel\n");
    if(threadid<dataSize){

        int idx = atomicAdd(index,1);

        if(idx < bufferSize/2)
        {
            buffer1[idx + *current_buffer] = input[(*count)*bufferSize/2 + idx + *current_buffer];
            printf("inside the kernel %d\n",idx);
        }
        else if(idx == bufferSize/2)
        {   
            *isfull = 1;

            (*count)++;

            if(*current_buffer != 0)
            {
                *current_buffer = 0;
            }
            else
            {
                *current_buffer = bufferSize/2;
            }
            idx = 0;
            *index = 1;

            buffer1[idx + *current_buffer] = input[(*count)*bufferSize/2 + idx + *current_buffer];
        }
        else
        {
            while(*index >= bufferSize/2)
            {
                ;
            }

            idx = atomicAdd(index,1);
            buffer1[idx + *current_buffer] = input[(*count)*bufferSize/2 + idx + *current_buffer];
        }
    }

}

void copy_thread(int *buffer1, int* output, int bufferSize, int *isfull)
{
    int count = 0;
    int current_buffer = 0;

    // create cuda stream to copy data from buffer to output
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    //cout<<"created thread \n";
    while(true)
    {
        if(*isfull == 1)
        {
            cout<<"inside the thread\n";
            cudaMemcpyAsync(output + (count)*bufferSize/2, buffer1 + current_buffer, (bufferSize/2)*sizeof(int), cudaMemcpyDeviceToHost,stream);
            cudaDeviceSynchronize();
            cout<<"inside the thread 22\n";
            
            if(current_buffer == 0)
            {
                current_buffer = bufferSize/2;
            }
            else
            {
                current_buffer = 0;
            }
            *isfull = 0;
            count++;
            
        }
    }
}


int main(){
    int dataSize = 1e3;
    int bufferSize = 1e2;

    int *buffer1,*buffer2,*offset,*currentOffset,*inputDev,*count, *isfull;

    int *input,*output;

    input = (int*)malloc(dataSize*sizeof(int));
    output = (int*)malloc(dataSize*sizeof(int));

    for(int i = 0; i<dataSize; i++){
        input[i] = i;
    }

    cudaMalloc(&buffer1,bufferSize*sizeof(int));
    cudaMalloc(&buffer2,bufferSize*sizeof(int));
    cudaHostAlloc(&isfull,sizeof(int),cudaHostAllocDefault);

    cudaMalloc(&offset,sizeof(int));
    cudaMalloc(&count,sizeof(int));
    cudaMalloc(&currentOffset,sizeof(int));

    cudaMalloc(&inputDev,dataSize*sizeof(int));

    cudaMemcpy(inputDev,input,dataSize*sizeof(int),cudaMemcpyHostToDevice);
    
    cudaMemset(offset,0,sizeof(int));
    cudaMemset(count,0,sizeof(int));
    cudaMemset(currentOffset,0,sizeof(int));
    cudaMemset(isfull,0,sizeof(int));
    cudaMemset(buffer1,-1,bufferSize*sizeof(int));

    // spawn a thread to copy data from buffer to output
    cout<<*isfull<<endl;
    thread t1(copy_thread,buffer1,output,bufferSize,isfull);
    t1.detach();
    
    write_output<<<dataSize/32+1,32>>>(inputDev,output,buffer1,offset,currentOffset,count,isfull,bufferSize,dataSize);
    // print cuda last error
    cout<<cudaGetLastError()<<endl;
    cudaDeviceSynchronize();

    for(int i = 0; i<dataSize; i++){
        cout<<output[i]<<" ";
    }

    cudaFree(buffer1);
    cudaFree(buffer2);
    cudaFree(offset);
    cudaFree(count);
    cudaFree(currentOffset);
    cudaFree(inputDev);
    cudaFreeHost(isfull);
    free(input);
    free(output);
}