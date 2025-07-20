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


__global__ void block_join(table *R, int sizeR, table *S, int sizeS, table *res)
{
    table current = R[threadIdx.x];
    int tid = threadIdx.x;
 
    for(int i = 0; i<sizeS; i++)
    {
        if(current.a == S[i].a)
        {
            res[tid * 10 + i].id = current.id;
            res[tid * 10 + i].a = current.a;
        }
    }
}

// __global__ void block_join(table *R, int sizeR, table *S, int sizeS,
//                            table *res) {
//   table current = R[threadIdx.x];
//   int tid = threadIdx.x;

//   for (int i = 0; i < sizeS; i++) {
//     if (current.a == S[i].a) {
//       res[tid * 10 + i].id = current.id;
//       res[tid * 10 + i].a = current.a;
//     }
//   }
// }

void block_join_utility() {
  vector<table> R, S;
  vector<pair<int, int>> result;
  int sizeS, sizeR;
  table *dev_R, *dev_S, *dev_res, *res;

  load_data(R, "r_table.csv");
  load_data(S, "s_table.csv");

  table *RR = &R[0];
  table *SS = &S[0];
  sizeR = R.size();
  sizeS = S.size();

  res = (table *)malloc(100 * sizeof(table));

  cudaMalloc((void **)&dev_R, 10 * sizeof(table));
  cudaMalloc((void **)&dev_S, 10 * sizeof(table));
  cudaMalloc((void **)&dev_res, 100 * sizeof(table));

  for (int i = 0; i < sizeS; i += 10) {
    cudaMemcpy(dev_S, SS + i, 10 * sizeof(table), cudaMemcpyHostToDevice);

    for (int j = 0; j < sizeR; j += 10) {
      cudaMemcpy(dev_R, RR + j, 10 * sizeof(table), cudaMemcpyHostToDevice);
      block_join<<<1, 10>>>(dev_R, 10, dev_S, 10, dev_res);
      cudaMemcpy(res, dev_res, 100 * sizeof(table), cudaMemcpyDeviceToHost);
      cudaMemset(dev_res, 0, 100 * sizeof(table));
      for (int k = 0; k < 100; k++) {
        if (res[k].id) {
          result.push_back({res[k].id, res[k].a});
        }
      }
    }
  }

  sort(result.begin(), result.end());

  for (int i = 0; i < result.size(); i++)
    cout << "id : " << result[i].first << " a : " << result[i].second << endl;
}

int main() {
  block_join_utility();
  return 0;
}
