#include "string_hash.h"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "linearprobing.h"


void basic_join_util(vector<Tuple> &data,vector<KeyValue> &hashedData, string file) {
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


__global__ void simple_join(KeyValue *R, int sizeR, KeyValue *S, int sizeS, int* index, Result* resDev) {

  int threadid = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadid< sizeR){
        KeyValue current = R[threadid];
        for (int i = 0; i < sizeS; i++) {
            if (current.key == S[i].key){
                int idx = atomicAdd(index, 1);
                resDev[idx].rid = current.value;
                resDev[idx].sid = S[i].value;
            }
        }
    }
}

int main(){
    vector<Tuple> dataR, dataS;
	vector<KeyValue> hashedDataR, hashedDataS;
    Result *res;
	int* index;
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    string dir = "/home/naruto/Documents/DBMS_Project/data/";
 
    basic_join_util(dataR, hashedDataR,dir+"table_r3.csv");
    basic_join_util(dataS, hashedDataS,dir+"table_s3.csv");

    int sizeR = hashedDataR.size();
    int sizeS = hashedDataS.size();
    int sizeResult = 51000000;
    
    KeyValue* hashedDataRDev = &hashedDataR[0];
    KeyValue* hashedDataSDev = &hashedDataS[0];
    Result* resDev;

    start = std::chrono::high_resolution_clock::now();
    cudaMalloc(&hashedDataRDev, sizeof(KeyValue) * sizeR);
    cudaMalloc(&hashedDataSDev, sizeof(KeyValue) * sizeS);
    cudaMalloc(&index, sizeof(int));
    cudaHostAlloc(&resDev, sizeof(Result) * sizeResult,cudaHostAllocDefault);
    
    cudaMemset(resDev, -1, sizeof(Result) * sizeResult);
    cudaMemset(index, 0, sizeof(int));
    
    cudaMemcpy(hashedDataRDev, hashedDataR.data(), sizeR * sizeof(KeyValue), cudaMemcpyHostToDevice);
    cudaMemcpy(hashedDataSDev, hashedDataS.data(), sizeS * sizeof(KeyValue), cudaMemcpyHostToDevice);

    simple_join<<<max(sizeS/32,1), 32>>>(hashedDataRDev, sizeR, hashedDataSDev, sizeS, index, resDev);

    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();

    cout << "Time taken to execute join: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() *1e-6<< " milliseconds" << endl;
    // res = (Result*) malloc(sizeResult * sizeof(Result));
    // Copy data from GPU to CPU
    // cudaMemcpy(res, resDev, sizeResult * sizeof(Result), cudaMemcpyDeviceToHost);

    // Find number of non-empty key/values in result array
    int numResults = 0;
    for (int i = 0; i < sizeResult; i++) {
        if (resDev[i].rid != -1 && resDev[i].sid != -1) {
            int rid = resDev[i].rid;
            int sid = resDev[i].sid;
            if(dataR[rid-1].a_value == dataS[sid-1].a_value)
                numResults++;
            else{
                // printf("Index: %d\n", i);
                // printf("Error: %d: %s\n %d:%s\n Hash: %u\n", rid, dataR[rid-1].a_value.c_str(), sid,dataS[sid-1].a_value.c_str(),resDev[i].a);
                // cout<<dataR[rid-1].a_value<<" "<<dataS[sid-1].a_value<<" Hash: "<< res[i].a<<endl;
            }
        }
    }

    // Print out the results    
    cout << "Number of results: " << numResults << endl;


    cudaFree(hashedDataRDev);
    cudaFree(hashedDataSDev);
}