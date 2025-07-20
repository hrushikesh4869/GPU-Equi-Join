#include "string_hash.h"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "linearprobing.h"
 
using namespace std;
 
// struct Result
// {
//     int rid;
//     int sid;
// };
 
__device__ __forceinline__  uint32_t hash_re(uint32_t k) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity - 1);
}
 
void hash_join_util(vector<Tuple> &data,vector<KeyValue> &hashedData, string file) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    load_data(data, file);
    end = std::chrono::high_resolution_clock::now();
    cout << "Time taken to load data: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()<< " nanoseconds" << endl;
 
    // count time taken to generate hash
 
    start = std::chrono::high_resolution_clock::now();
    generate_hash(data, hashedData);
    end = std::chrono::high_resolution_clock::now();
    cout << "Time taken to generate hash: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()<< " nanoseconds" << endl;
}
 
__global__ void gpu_hash_join(KeyValue *hashTable,KeyValue* hashedDataS, uint32_t sizeS, uint32_t sizeR, Result *result ,int *count){
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
 
    if(threadid< sizeS){
        uint32_t key = hashedDataS[threadid].key;
        uint32_t slot;
        
        slot = hash_re(key);
        while (true) {
            
            if (hashTable[slot].key == key) {
                int idx = atomicAdd(count,1);
                result[idx].rid = hashTable[slot].value;
                result[idx].sid = hashedDataS[threadid].value;
            }
            if (hashTable[slot].key == kEmpty) {
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}
 
int main() {
 
    vector<Tuple> dataR, dataS;
    vector<KeyValue> hashedDataR, hashedDataS;
    int *count, sizeResult = 5100000, numResults = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    Result *result = (Result*) malloc(sizeof(Result) * sizeResult);
 
    hash_join_util(dataR, hashedDataR,"table_r2.csv");
    hash_join_util(dataS, hashedDataS,"table_s2.csv");
 
    int sizeR = hashedDataR.size();
    int sizeS = hashedDataS.size();
    
    start = std::chrono::high_resolution_clock::now();
    KeyValue* hashedDataRDev = &hashedDataR[0];
    KeyValue* pHashTable = create_hashtable();
    insert_hashtable(pHashTable, hashedDataRDev, sizeR);
    
 
    KeyValue* hashedDataSDev;
 
    //cudaHostAlloc(&hashedDataSDev, sizeof(KeyValue) * sizeS, cudaHostAllocDefault);
    cudaMalloc(&hashedDataSDev, sizeof(KeyValue) * sizeS);

    // cudaHostAlloc(&result, sizeof(Result) * sizeResult,cudaHostAllocDefault);
    Result* resultDev;
    cudaMalloc(&resultDev, sizeof(Result) * sizeResult);

    // // check if the allocation was successful
    // if (result == NULL)
    // {
    //     printf("Failed to allocate memory!\n");
    //     exit(EXIT_FAILURE);
    // }

    cudaMalloc(&count,sizeof(int));
 
    cudaMemset(count,0,sizeof(int));
    cudaMemset(result, -1, sizeof(Result) * sizeResult);
    cudaMemcpy(hashedDataSDev, hashedDataS.data(), sizeof(KeyValue) * sizeS, cudaMemcpyHostToDevice);
    

    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize,gpu_hash_join, 0, 0);
    cout<<mingridsize<<" "<<threadblocksize<<endl;
 
    gpu_hash_join<<<max(sizeS/threadblocksize+1,1), threadblocksize>>>(pHashTable, hashedDataSDev, sizeS, sizeR, resultDev, count);
    
    // gpu_hash_join<<<max(sizeS/32,1), 32>>>(pHashTable, hashedDataSDev, sizeS, sizeR, resultDev, count);


    cudaMemcpy(result, resultDev, sizeof(Result) * sizeResult, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    int total=0;
    cudaMemcpy(&total, count, sizeof(int), cudaMemcpyDeviceToHost);

    end = std::chrono::high_resolution_clock::now();
 
    cout << "Time taken to do the join: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1e-6<< " milliseconds" << endl;
    
    cout<<total<<endl;
    for(long int i = 0; i<sizeResult && i<total; i++)
    {
        if(result[i].rid != -1 && result[i].sid != -1 )
        {
            int rid = result[i].rid;
            int sid = result[i].sid;
            if(dataR[rid-1].a_value == dataS[sid-1].a_value)
                numResults++;
        }
    }
 
    cout<<numResults<<endl;
 
    cudaFree(count);
    cudaFreeHost(resultDev);
    cudaFree(hashedDataSDev);
 
  return 0;
}