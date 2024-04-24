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
#include <memory.h>
#include <algorithm>

#include "MDP_type_definitions.h"
#include "pretty_printing_MDP.h"
#include "MDP_generation.h"
#include "VI_algorithms_helper_methods.h"
#include "VI_algorithm.h"
#include "BVI_algorithm.h"
#include "VIAE_algorithm.h"
#include "VIAEH_algorithm.h"
#include "VIH_algorithm.h"
#include "experiments.h"

using namespace std;
using namespace std::chrono;

V_type value_iteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 2. Improved Lower Bound
	double **V = new double *[2];
	for (int i = 0; i < 2; ++i)
	{
		V[i] = new double[S];
	}
	int siz = sqrt(S - 1) - 2;
	int Xmax = siz + 2;
	siz = Xmax / 2;
	for (int s = 0; s < S; s++)
	{
		// V[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];

		int x_curr = s % Xmax;
		int y_curr = s / Xmax;

		double x1 = sqrt(pow(abs(x_curr - siz), 2) + pow(abs(y_curr - siz), 2));
		// V[0][s] = 0;
		V[1][s] = 0;
		// V[0][s] = 1;
		V[0][s] = -x1 * 5 - 10;
		// V[0][s] = -500;
	}
	V[0][S - 1] = 0;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// keep track of work done in each iteration in microseconds
	// start from iteration 1
	vector<microseconds> work_per_iteration(1);

	// keep count of number of iterations
	int iterations = 0;
	const double convergence_bound_precomputed = 0.0005;
	gamma = 1;
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	while (abs_max_diff(V[0], V[1], S) > (convergence_bound_precomputed))
	{

		// Increment iteration counter i
		iterations++;
		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If i is even, then (i & 1) is 0, and the one to change is V[0]
		double *V_current_iteration = V[(iterations & 1)];
		double *V_previous_iteration = V[1 - (iterations & 1)];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			// V_current_iteration[s] = numeric_limits<double>::min();
			V_current_iteration[s] = -1000000;

			// ranged for loop over all actions in the action set of state s
			for (auto a : A[s])
			{
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];
				double R_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_previous_iteration, P_s_a_nonzero);
				if (R_s_a > V_current_iteration[s])
				{
					V_current_iteration[s] = R_s_a;
				}
			}
		}

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
		/*for (int i = 0; i < S; ++i) {
			if(V_previous_iteration[i] - V_current_iteration[i]	>0)
}*/
	}
	vector<double> result(V[(iterations & 1)], V[(iterations & 1)] + S);
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE MEMORY
	for (int i = 0; i < 2; ++i)
	{
		delete[] V[i];
	}
	delete[] V;

	return result_tuple;
}


class MBIE {
	public:
	//delta = 0.05;
	int m;//; = 100;
	int nA;// = 4;
	int nS;
	int gamma;
	// max(A[0].size();)
	// or S*4;
	double delta;// = delta / (2 * S * nA * m);
	int s_state;// = 0;
	int **Nsa;// = NULL;
	double ***hatP;// = NULL;
	double **Rsa;// = NULL;
	int ***Nsas;// = NULL;
	double **hatR;// = NULL;
	double **confR;// = NULL;
	double **confP;// = NULL;
	vector<double> max_p;

	int current_s;
	int last_action;

	MBIE(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon, double delta, int m) {
		delta = 0.05;
		m = 1000;
		nA = 4;
		nS = S;
		gamma = gamma;
		// max(A[0].size();)
		// or S*4;
		delta = delta / (2 * S * nA * m);
		int s_state = 0;
		hatP = new double **[S];
		Nsas = new int **[S];
		Nsa = new int *[S];
		Rsa = new double *[S];
		hatR = new double *[S];
		confR = new double *[S];
		confP = new double *[S];

		vector<double> max_p(nS, 0.0);
		current_s = 0;
		last_action = -1;
		
		for (int i = 0; i < S; ++i)
		{
			Nsa[i] = new int[nA];
			Rsa[i] = new double[nA];
			hatR[i] = new double[nA];
			confR[i] = new double[nA];
			confP[i] = new double[nA];
			memset(Nsa[i], 0, sizeof(int) * nA);
			memset(Rsa[i], 0, sizeof(double) * nA);
			memset(hatR[i], 0, sizeof(double) * nA);
		}

		//
		//	self.confP = np.zeros((self.nS, self.nA))

		for (int i = 0; i < S; i++)
		{
			Nsas[i] = new int *[nA];
			hatP[i] = new double *[nA];
			for (int j = 0; j < nA; j++)
			{
				Nsas[i][j] = new int[S];
				hatP[i][j] = new double[S];
				memset(Nsas[i][j], 0, sizeof(int) * S);
				memset(hatP[i][j], 0, sizeof(double) * S);
			}
		}
	}
	void confidence();
	void reset(S_type init);
	void max_proba(vector<int> sorted_indices, int s, int a);
	vector<int> EVI(double epsilon);
	void play(S_type S, R_type R);
};

void MBIE::play(S_type S, R_type R) {

}

void MBIE::confidence()
{
	for (int s = 0; s < nS; s++)
	{
		for (int a = 0; a < nA; a++)
		{
			confP[s][a] = sqrt((2 * (log(pow(nS, 2) - 2) - log(delta)) / max(1, Nsa[s][a])));
			confR[s][a] = sqrt(log(2 / delta) / (2 * max(1, Nsa[s][a])));
		}
	}
}

void MBIE::reset(S_type init)
{
	for (int i = 0; i < nS; i++)
	{
		for (int j = 0; j < nA; j++)
		{
			Rsa[i][j] = 0;
			hatR[i][j] = 0;
			confR[i][j] = 0;
			confP[i][j] = 0;
			for (int k = 0; k < nS; k++)
			{
				Nsas[i][j][k] = 0;
				hatP[i][j][k] = 0;
			}
		}
	}
	//TODO: init start and last
}

void MBIE::max_proba(vector<int> sorted_indices, int s, int a)
{
	double min1 = min(1.0, hatP[s][a][sorted_indices[s - 1]] + confP[s][a] / 2);
	
	std::fill(max_p.begin(),max_p.end(),0.0);
	
	int l = 0;

	if (min1 == 1)
	{
		max_p[sorted_indices[nS - 1]] = 1;
	}
	else
	{	
		//Mega copy hack (that may or may not work)
		max_p.assign(*hatP[s], *hatP[s] + nA);
		//vector<double> max_p(hatP[s][a].begin(), hatP[s][a].end());
		max_p[sorted_indices[nS - 1]] += confP[s][a] / 2.0;
		l = 0;
		double sum_max_p = 0.0;
		for (size_t i = 0; i < max_p.size(); ++i)
		{
			sum_max_p += max_p[i];
		}
		while (sum_max_p > 1.0)
		{
			max_p[sorted_indices[l]] = max(0.0, 1.0 - sum_max_p + max_p[sorted_indices[l]]);
			++l;

			// Recalculate the sum of max_p
			sum_max_p = 0.0;
			for (size_t i = 0; i < max_p.size(); ++i)
			{
				sum_max_p += max_p[i];
			}
		}
		
	}
	//max_p has been set
}

vector<int> MBIE::EVI(double epsilon)
{
	int niter = 0;
	//int nS = S;
	vector<int> sorted_indices(nS);
	
	int nA = 4;
	// Fill the vector with indices
	iota(sorted_indices.begin(), sorted_indices.end(), 0);
	vector<int> policy(nS, 0);
	std::vector<double> V0(nS);
	for (int i = 0; i < nS; ++i)
	{
		V0[i] = 1.0 / (1.0 - gamma);
	}

	// Initialize V1
	vector<double> V1(nS, 0.0); // Initialize with zeros
	epsilon = epsilon * (1 - gamma) / (2 * gamma);
	double R_s_a=0;

	while (true)
	{
		for (int s = 0; s < nS; s++)
		{
			for (int a = 0; a < nA; a++)
			{
				max_proba(sorted_indices, s, a);
				//auto &[P_s_a, P_s_a_nonzero] = hatP[s][a];
				double R_s_a = hatR[s][a] + confR[s][a] + gamma * sum_of_mult(max_p, V0);
				if (a == 0 || R_s_a > V1[s])
				{
					V1[s] = R_s_a;
					policy[s] = a;
				}
			}
		}
		// V distance
		int dist = 0;
		for (int i = 0; i < nS; i++) 
		{
			dist += (V0[i]-V1[i])*(V0[i]-V1[i]);
		}
		dist = sqrt(dist);
		
		if (dist < epsilon) 
		{
			return policy;
		} 
		else 
		{
			V0 = V1; //copy
			for (int i = 0; i < nS; ++i)
			{
				V1[i] = 1.0 / (1.0 - gamma);
			}
			//sorted indices
			iota(sorted_indices.begin(), sorted_indices.end(), 0);
			sort(sorted_indices.begin(), sorted_indices.end(), [&](int i,int j){return V0[i]<V0[j];} );
		}
		// max_iter break would go here 
	}
}
/*V_type MBIE(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon, double delta, int m)
{
	delta = 0.05;
	m = 100;
	int nA = 4;
	// max(A[0].size();)
	// or S*4;
	delta = delta / (2 * S * nA * m);
	int s_state = 0;
	int **Nsa = NULL;
	float ***hatP = NULL;
	float **Rsa = NULL;
	int ***Nsas = NULL;
	float **hatR = NULL;
	float **confR = NULL;
	float **confP = NULL;
	hatP = new float **[S];
	Nsas = new int **[S];
	Nsa = new int *[S];
	Rsa = new float *[S];
	hatR = new float *[S];
	confR = new float *[S];
	confP = new float *[S];

	for (int i = 0; i < S; ++i)
	{
		Nsa[i] = new int[nA];
		Rsa[i] = new float[nA];
		hatR[i] = new float[nA];
		confR[i] = new float[nA];
		confP[i] = new float[nA];
		memset(Nsa[i], 0, sizeof(int) * nA);
		memset(Rsa[i], 0, sizeof(float) * nA);
		memset(hatR[i], 0, sizeof(float) * nA);
	}

	//
	//	self.confP = np.zeros((self.nS, self.nA))

	for (int i = 0; i < S; i++)
	{
		Nsas[i] = new int *[nA];
		hatP[i] = new float *[nA];
		for (int j = 0; j < nA; j++)
		{
			Nsas[i][j] = new int[S];
			hatP[i][j] = new float[S];
			memset(Nsas[i][j], 0, sizeof(int) * S);
			memset(hatP[i][j], 0, sizeof(float) * S);
		}
	}
}*/

V_type value_iterationGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 2. Improved Lower Bound
	double **V = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V[i] = new double[S];
	}
	// int siz=sqrt(S-1)-2;
	// int Xmax=siz+2;
	// gamma=1;
	for (int s = 0; s < S; s++)
	{
		/*
		int x_curr=s%Xmax;
			int y_curr=s/Xmax;
		double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
		V[0][s] = -x1*5-10;*/
		V[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
	} // V[0][S-1] = 0;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// keep track of work done in each iteration in microseconds
	// start from iteration 1
	vector<microseconds> work_per_iteration(1);

	// keep count of number of iterations
	int iterations = 0;
	bool upper_convergence_criteria = false;
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double convergence_bound_precomputed = 0.0005;

	while (!upper_convergence_criteria)
	{
		upper_convergence_criteria = true;
		// Increment iteration counter i
		iterations++;
		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If i is even, then (i & 1) is 0, and the one to change is V[0]
		double *V_current_iteration = V[0];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			// V_current_iteration[s] = double(0);
			double oldV = V_current_iteration[s];
			// ranged for loop over all actions in the action set of state s
			for (auto a : A[s])
			{
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];
				double R_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_current_iteration, P_s_a_nonzero);
				if (R_s_a > V_current_iteration[s])
				{
					V_current_iteration[s] = R_s_a;
				}
			}
			if (abs(oldV - V_current_iteration[s]) > convergence_bound_precomputed)
				upper_convergence_criteria = false;
		}

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}
	vector<double> result(V[0], V[0] + S);
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE MEMORY
	for (int i = 0; i < 1; ++i)
	{
		delete[] V[i];
	}
	delete[] V;

	return result_tuple;
}

V_type value_iterationGSTM(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon, int D3)
{

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 2. Improved Lower Bound
	double **V = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V[i] = new double[S];
	}
	// int siz=sqrt(S-1)-2;
	// int Xmax=siz+2;
	// gamma=1;
	for (int s = 0; s < S; s++)
	{
		/*
		int x_curr=s%Xmax;
			int y_curr=s/Xmax;
		double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
		V[0][s] = -x1*5-10;*/
		if (D3 == 0)
			V[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
		else
		{
			// int x_curr=s%Xmax;
			// int y_curr=s/Xmax;
			// double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
			V[0][s] = -500;
			// V[0][s] = -x1*5-10;
			gamma = 1;
		}
	}
	if (D3 != 0)
		V[0][S - 1] = 0;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// keep track of work done in each iteration in microseconds
	// start from iteration 1
	vector<microseconds> work_per_iteration(1);

	// keep count of number of iterations
	int iterations = 0;
	bool upper_convergence_criteria = false;
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double convergence_bound_precomputed = 0.0005;

	while (!upper_convergence_criteria)
	{
		upper_convergence_criteria = true;
		// Increment iteration counter i
		iterations++;
		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If i is even, then (i & 1) is 0, and the one to change is V[0]
		double *V_current_iteration = V[0];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			// V_current_iteration[s] = double(0);
			double oldV = V_current_iteration[s];
			// ranged for loop over all actions in the action set of state s
			for (auto a : A[s])
			{
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];
				double R_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_current_iteration, P_s_a_nonzero);
				if (R_s_a > V_current_iteration[s])
				{
					V_current_iteration[s] = R_s_a;
				}
			}
			if (abs(oldV - V_current_iteration[s]) > convergence_bound_precomputed)
				upper_convergence_criteria = false;
		}

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}
	vector<double> result(V[0], V[0] + S);
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE MEMORY
	for (int i = 0; i < 1; ++i)
	{
		delete[] V[i];
	}
	delete[] V;

	return result_tuple;
}