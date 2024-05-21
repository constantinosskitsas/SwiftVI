#ifndef VI_ALGORITHM_H
#define VI_ALGORITHM_H

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

#include "MDP_type_definitions.h"

using namespace std;

class MBIE {
	public:
	//delta = 0.05;
	int m;//; = 100;
	int nA;// = 4;
	int nS;
	double gamma;
	double epsilon;
	vector<int> policy;
	// max(A[0].size();)
	// or S*4;
	double delta;// = delta / (2 * S * nA * m);
	int s_state;// = 0;
	int *StateSwift; //false-> do VI, true do SwiftVI
	int **Nsa;// = NULL;
	double ***hatP;// = NULL;
	double **Rsa;// = NULL;
	int ***Nsas;// = NULL;
	double **hatR;// = NULL;
	double **confR;// = NULL;
	double **confP;// = NULL;
	vector<double> max_p;
	int cnt = 0;

	int current_s;
	int last_action;

	MBIE(S_type S, int nA, double gamma, double epsilon, double delta, int m); 
	//void parallel_fill(int s, int a);
	//void fill_segment(int s, int a, int start, int end);
	std::tuple<int,std::vector<int>> update_vals(int state, double reward);
	void confidence();
	void reset(S_type init);
	void max_proba(vector<int> sorted_indices, int s, int a);
	vector<int> EVI();
    vector<int> swiftEVI();
	std::tuple<int,std::vector<int>> play(int state, double reward);
    std::tuple<int,std::vector<int>> playswift(int state, double reward);
};



V_type value_iteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iterationGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon);
V_type value_iterationGSTM(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon,int D3);

#endif
