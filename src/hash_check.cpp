#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "linearprobing.h"
#include <city.h>

using namespace std;

uint32_t hash_re(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity - 1);
}

int main()
{
    uint32_t nu1 = 1382933766;
    uint32_t nu2 = 3490885298;

    cout << hash_re(nu1) << endl;
    cout << hash_re(nu2) << endl;

    string s1 = "belis";
    string s2 = "dunting";

    uint32_t res1 = CityHash32(s1.c_str(), s1.size());
    uint32_t res2 = CityHash32(s2.c_str(), s2.size());

    cout << res1 << " " << res2 << endl;
}