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
    cout << "Time taken to generate hash: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1e-6<< " milliseconds" << endl;
}
 
 
__global__ void compute_upperbound(KeyValue *hashTable,KeyValue* hashedDataS, uint32_t sizeS, uint32_t sizeR, unsigned long long int *upperbound){
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
 
    if(threadid< sizeS){
        uint32_t key = hashedDataS[threadid].key;
        uint32_t slot;
        
        slot = hash_re(key);
        while (true) {
            
            if (hashTable[slot].key == key) {
                long int idx = atomicAdd(upperbound,1);
            }
            if (hashTable[slot].key == kEmpty) {
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
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
 
Result* pinned_hash_join(KeyValue *pHashTable, KeyValue *hashedDataSDev, int sizeS, int sizeR, unsigned long long int sizeResult, int &total)
{
    Result *result;
    int *count;
 
    cudaHostAlloc(&result, sizeof(Result) * sizeResult,cudaHostAllocDefault);
    cudaMemset(result, -1, sizeof(Result) * sizeResult);
     
    cudaMalloc(&count,sizeof(int));
    cudaMemset(count,0,sizeof(int));
 
    gpu_hash_join<<<max(sizeS/32,1), 32>>>(pHashTable, hashedDataSDev, sizeS, sizeR, result, count);
 
    cout<<cudaGetErrorString(cudaGetLastError())<<endl;
    //cudaMemcpy(result, resultDev, sizeof(Result) * sizeResult, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
 
    cudaMemcpy(&total, count, sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}
 
Result *normal_hash_join(KeyValue *pHashTable, KeyValue *hashedDataSDev, int sizeS, int sizeR, unsigned long long int sizeResult, int &total)
{
    Result *result, *resultDev;
    int *count;
 
    result = (Result *)malloc(sizeof(Result) * sizeResult);
 
    cudaMalloc(&resultDev, sizeof(Result) * sizeResult);
    cudaMemset(resultDev, -1, sizeof(Result) * sizeResult);
 
    cudaMalloc(&count,sizeof(int));
    cudaMemset(count,0,sizeof(int));
 
    gpu_hash_join<<<max(sizeS/32,1), 32>>>(pHashTable, hashedDataSDev, sizeS, sizeR, resultDev, count);
    cudaMemcpy(result, resultDev, sizeof(Result) * sizeResult, cudaMemcpyDeviceToHost);
 
    cudaMemcpy(&total, count, sizeof(int), cudaMemcpyDeviceToHost);
 
    return result;
 
}
 
int main() {
 
    vector<Tuple> dataR, dataS;
    vector<KeyValue> hashedDataR, hashedDataS;
    unsigned long long int sizeResult = 0, numResults = 0;
    int total = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    
    string dir = "/home/naruto/Documents/DBMS_Project/data/";
 
    hash_join_util(dataR, hashedDataR,dir+"table_r9.csv");
    hash_join_util(dataS, hashedDataS,dir+"table_s9.csv");
 
    int sizeR = hashedDataR.size();
    int sizeS = hashedDataS.size();
    
    // unsigned long long reserve_memory = 5*(1UL<<30);
    // void* ptr{nullptr};
    // cudaMalloc(&ptr, reserve_memory);
    
    start = std::chrono::high_resolution_clock::now();
    KeyValue* hashedDataRDev = &hashedDataR[0];
    KeyValue* pHashTable = create_hashtable();
    insert_hashtable(pHashTable, hashedDataRDev, sizeR);
    
 
    KeyValue* hashedDataSDev;
 
    cudaMalloc(&hashedDataSDev, sizeof(KeyValue) * sizeS);
    cudaMemcpy(hashedDataSDev, hashedDataS.data(), sizeof(KeyValue) * sizeS, cudaMemcpyHostToDevice);
    
 
    
    unsigned long long int *upperbound;
    cudaMalloc(&upperbound,sizeof(unsigned long long int));
    cudaMemset(upperbound,0,sizeof(unsigned long long int));
    
    compute_upperbound<<<max(sizeS/32,1), 32>>>(pHashTable, hashedDataSDev, sizeS, sizeR, upperbound);
    
    cudaMemcpy(&sizeResult, upperbound, sizeof(int), cudaMemcpyDeviceToHost);
    
    cout<<sizeResult<<endl;
 
    size_t free_byte;
    size_t total_byte ;
 
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
 
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }
 
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    Result *result;
 
    printf("GPU memory usage total = %fB, free = %fB \n",total_db,free_db);
 
    if(free_db > sizeResult * 8 && false)
    {   cout<<"Normal memory"<<endl;
        result = normal_hash_join(pHashTable, hashedDataSDev,sizeS,sizeR,sizeResult,total);
    }
    else
    {   cout<<"Pinned memory"<<endl;
        result = pinned_hash_join(pHashTable, hashedDataSDev,sizeS,sizeR,sizeResult,total);
    }
 
    end = std::chrono::high_resolution_clock::now();
 
    cout << "Time taken to do the join: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1e-6<< " milliseconds" << endl;
 
    start = std::chrono::high_resolution_clock::now();
    
    materialize_results(dataR, dataS,result, sizeResult,total);
 
    end = std::chrono::high_resolution_clock::now();
 
    cout << "Time taken to materialize the join: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()*1e-6<< " milliseconds" << endl;
 
    //cudaFree(count);
    cudaFreeHost(result);
    cudaFree(hashedDataSDev);
 
  return 0;
}