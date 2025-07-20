#include "city.h"
#include <bits/stdc++.h>
#include <thread>
#include "linearprobing.h"
using namespace std;

int numthreads = 8;

// Create a structure to hold key/value pairs
struct Tuple
{
  uint32_t key;
  string a_value;
};

struct Result
{
  int rid;
  int sid;
};

void load_data(vector<Tuple> &data, string file)
{
  string mytext, word;
  Tuple temp;
  ifstream readme(file);

  while (getline(readme, mytext))
  {
    stringstream ss(mytext);

    getline(ss, word, ',');
    temp.key = stoi(word);

    getline(ss, word, ',');
    temp.a_value = word;

    data.push_back(temp);
  }
}

void calculate_hash(vector<Tuple> &data, vector<KeyValue> &hashedData, int start, int end)
{
  int size = data.size();
  for (int i = start; i < end; i++)
  {
    hashedData[i].value = data[i].key;
    hashedData[i].key = CityHash32(data[i].a_value.c_str(), data[i].a_value.size());
  }
}

void generate_hash(vector<Tuple> &data, vector<KeyValue> &hashedData)
{

  int size = data.size();
  int blocksize = size / numthreads;
  vector<thread> threads;

  hashedData = vector<KeyValue>(size);

  for (int i = 0; i < numthreads; i++)
  {
    int start = i * blocksize;
    int end = (i == numthreads - 1) ? size : (i + 1) * blocksize;
    threads.push_back(thread(calculate_hash, ref(data), ref(hashedData), start, end));
  }

  for (int i = 0; i < numthreads; i++)
    threads[i].join();
}

void materialize_results_th(vector<Tuple> &dataR, vector<Tuple> &dataS, Result *result, vector<int> &count, int start, int end, int total, int th_idx)
{
  int numResults = 0;
  for (long int i = start; i < end && i < total; i++)
  {
    if (result[i].rid != -1 && result[i].sid != -1)
    {
      int rid = result[i].rid;
      int sid = result[i].sid;
      if (dataR[rid - 1].a_value == dataS[sid - 1].a_value)
        numResults++;
    }
  }
  count[th_idx] = numResults;
}

void materialize_results(vector<Tuple> &dataR, vector<Tuple> &dataS, Result *result, int size, int total)
{
  int blocksize = size / numthreads;
  vector<int> count(numthreads, 0);
  vector<thread> threads;
  int res_count = 0;

  for (int i = 0; i < numthreads; i++)
  {
    int start = i * blocksize;
    int end = (i == numthreads - 1) ? size : (i + 1) * blocksize;
    threads.push_back(thread(materialize_results_th, ref(dataR), ref(dataS), result, ref(count), start, end, total, i));
  }

  for (int i = 0; i < numthreads; i++)
  {
    threads[i].join();
    res_count += count[i];
  }

  cout << "Total results are : " << res_count << endl;
}