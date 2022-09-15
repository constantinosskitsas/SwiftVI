#ifndef MDP_TYPE_DEFINITIONS_H
#define MDP_TYPE_DEFINITIONS_H

#include <algorithm>
#include <chrono>
#include <tuple>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
using namespace std::chrono;

typedef vector<vector<double> > R_type;
typedef vector<vector<int> > A_type;

//S x A x (S, nonzero_states)
typedef vector<vector<pair<vector<double>, vector<int> > > > P_type;

//typedef vector<double> V_type;
typedef tuple<vector<double>, int, vector<microseconds>, vector<vector<pair<int, int>>>> V_type;

typedef vector<double> V_result_type;

typedef int S_type;
typedef pair<double, int> q_action_pair_type;
typedef vector<q_action_pair_type> heap_of_pairs_type;
typedef tuple<R_type, A_type, P_type> MDP_type;
#endif
