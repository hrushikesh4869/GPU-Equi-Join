#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

struct table {
  int id;
  int a;
};

void load_data(vector<table> &data, string file) {
  string mytext;
  table temp;
  vector<int> store;
  ifstream readme(file);
  long i = 0;
  while (getline(readme, mytext)) {
    stringstream ss(mytext);
    string word;
    store.clear();
    while (getline(ss, word, ',')) {
      store.push_back(stoi(word));
    }
    temp.id = store[0];
    temp.a = store[1];
    data.push_back(temp);
    i++;
  }

  readme.close();
}

__global__ void simple_join(table *R, int sizeR, table *S, int sizeS) {

  int i = threadIdx.x;
  table current = R[i];
  for (int i = 0; i < sizeS; i++) {
    if (current.a == S[i].a)
      printf("R and S join on a=%d and RID=%d, SID=%d\n", current.a, current.id,
             S[i].id);
  }
  //   printf("%d %d\n", R[i].id, R[i].a);
}

__global__ void block_simple_join(table *R, int sizeR, table *S, int sizeS,
                                  int *result, int *idx, int *mutex) {
  int i = threadIdx.x;
  table current = R[i];

  for (int i = 0; i < sizeS; i++) {
    if (current.a == S[i].a) {
      while (atomicExch(mutex, 1) != 0)
        ;
      result[*idx] = current.a;
      (*idx)++;
      atomicExch(mutex, 0);
    }
  }
}

void utility() {
  vector<table> R, S;
  int *d_mutex, *d_idx;
  int *arr = (int *)malloc(100 * sizeof(int));

  load_data(R, "r_table.csv");
  load_data(S, "s_table.csv");

  int sizeR, sizeS;
  sizeR = R.size();
  sizeS = S.size();
  table *RR = &R[0];
  table *SS = &S[0];
  table *d_R, *d_S;
  int *d_res;
  int k;

  cudaMalloc((void **)&d_R, 10 * sizeof(table));
  cudaMalloc((void **)&d_S, 10 * sizeof(table));
  cudaMalloc((void **)&d_res, 100 * sizeof(int));

  cudaMalloc((void **)&d_mutex, sizeof(int));
  cudaMalloc((void **)&d_idx, sizeof(int));

  cudaMemset(d_mutex, 0, sizeof(int));
  cudaMemset(d_idx, 0, sizeof(int));

  for (int i = 0; i < sizeS; i += 10) {
    cudaMemcpy(d_S, SS + i, 10 * sizeof(table), cudaMemcpyHostToDevice);
    cout << "copy data in d_s \n";
    for (int j = 0; j < sizeR; j += 10) {

      cudaMemcpy(d_R, RR + j, 10 * sizeof(table), cudaMemcpyHostToDevice);
      cudaMemset(d_mutex, 0, sizeof(int));
      cudaMemset(d_idx, 0, sizeof(int));

      block_simple_join<<<1, 10>>>(d_R, 10, d_S, 10, d_res, d_idx, d_mutex);

      cudaMemcpy(arr, d_res, 100 * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&k, d_idx, sizeof(int), cudaMemcpyDeviceToHost);

      for (int i = 0; i < k; i++)
        if (arr[i])
          cout << arr[i] << endl;
    }
  }
}

int main() {

  //   utility();
  //   return 0;

  vector<table> R, S;
  int *d_mutex, *d_idx;

  load_data(R, "r_table.csv");
  load_data(S, "s_table.csv");

  int sizeR, sizeS;
  sizeR = R.size();
  sizeS = S.size();
  table *RR = &R[0];
  table *SS = &S[0];
  table *d_R, *d_S;

  cudaMalloc((void **)&d_R, sizeR * sizeof(table));
  cudaMalloc((void **)&d_S, sizeS * sizeof(table));

  // Copy data from CPU to GPU
  cudaMemcpy(d_R, RR, sizeR * sizeof(table), cudaMemcpyHostToDevice);
  cudaMemcpy(d_S, SS, sizeS * sizeof(table), cudaMemcpyHostToDevice);

  // Launch the kernel
  simple_join<<<1, 1000>>>(d_R, sizeR, d_S, sizeS);

  // Free allocated memory on GPU
  cudaFree(d_R);
  cudaFree(d_S);
}