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
#include <cmath>

#include "AncVI-algorithm.h"
#include "VI_algorithms_helper_methods.h"

V_type anc_valueiteration(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 2. Improved Lower Bound
	double **V = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V[i] = new double[S];
	}

    // Define anchor
    double *Anchor = new double[S];

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
        Anchor[s] = V[0][s]; //Set anchor to V0 as presented in paper
	} // V[0][S-1] = 0;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// keep track of work done in each iteration in microseconds
	// start from iteration 1
	vector<microseconds> work_per_iteration(1);

	//policy
	vector<int> policy(S, 0);

	// keep count of number of iterations
	int iterations = 0;
	bool upper_convergence_criteria = false;
	const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	//const double convergence_bound_precomputed = 0.0005;
    double beta = 0;
    double beta_sum = 1; //Sum starts at 1 in round 0

	while (!upper_convergence_criteria)
	{
		upper_convergence_criteria = true;
		// Increment iteration counter i
		iterations++;

        //Calculate Anchor strengh (beta)
        beta_sum += pow(gamma, -2*iterations);
        beta = 1 / beta_sum;

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
				double R_s_a = beta*Anchor[s]+(1-beta)*(R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_current_iteration, P_s_a_nonzero));
				if (R_s_a > V_current_iteration[s])
				{
					V_current_iteration[s] = R_s_a;
					policy[s] = a;
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
/*
	std::cout << "V_star policy: ";
	for (int i: policy) {
		std::cout << i;
	}
	std::cout << std::endl;
*/
	// DEALLOCATE MEMORY
	for (int i = 0; i < 1; ++i)
	{
		delete[] V[i];
	}
	delete[] V;

    delete[] Anchor;

	return result_tuple;
}