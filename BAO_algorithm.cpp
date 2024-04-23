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

#include "heap_methods.h"

using namespace std;
using namespace std::chrono;

V_type value_iteration_BAO(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Finds out how big a vector we need to store an action in an entry
	// If the arrays has A_max size, then a = A_max has no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index
	int A_max = find_max_A(A) + 1;

	// Find relevant values from the R parameter
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[2];
	for (int i = 0; i < 2; ++i)
	{
		V_U[i] = new double[S];
	}
	// int siz=sqrt(S-1)-2;
	// int Xmax=siz+2;
	// siz=Xmax/2;
	for (int s = 0; s < S; s++)
	{
		// V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		/*int x_curr=s%Xmax;
		int y_curr=s/Xmax;

		int xa1= abs(x_curr-siz);
		int ya1= abs(y_curr-siz);
		double x2=0;
		if (xa1>ya1)
			x2=xa1;
		else
			x2=ya1;
		//double x1= sqrt( pow(xa1,2)+pow(ya1,2));
		//V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		*/
		V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s]; //+(x1*0.4) ;
		// V_U[0][s] = 0;
		// V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		// V_U[1][s] = 1.0;
		V_U[1][s] = 1.0;
	} // V_U[0][S-1] = 0;

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration as a dummy value
	vector<microseconds> work_per_iteration(1);

	// initialize criteria boolean variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged with the given criteria!
	bool upper_convergence_criteria = false;

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double convergence_bound_precomputed = 0.0005;
	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// arrays to keep the Q-values
	double **Q_values_per_state = new double *[S];
	for (int i = 0; i < S; ++i)
	{
		// Q_values_per_state[i] = new double[A[i].size()];
		Q_values_per_state[i] = new double[A_max];
	}

	//**********************************************************************

	for (int s = 0; s < S; s++)
	{
		// pointers to the heaps of current state s
		double *Q_values_s = Q_values_per_state[s];
		for (int a = 0; a < A_max; a++)
		{
			// for (int a = 0; a < (A[s].size()); a++){
			// Q_values_s[a] = (r_star_max / (1.0 - gamma));	//init with V_max as stated in BAO1 paper
			// Q_values_s[a] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s]; //init with upper bound V_U
			Q_values_s[a] = V_U[0][s];
			// Q_values_s[a]=1.0;
		}
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;
	// gamma=1;
	// while any of the criteria are NOT, !, met, run the loop
	while (!upper_convergence_criteria)
	{

		// Increment iteration counter
		iterations++;

		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If iiteration is even, then (iteration & 1) is 0, and the one to change is V[0]
		double *V_U_current_iteration = V_U[(iterations & 1)];
		double *V_U_previous_iteration = V_U[1 - (iterations & 1)];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			// keep best actions here
			double *Q_values_s = Q_values_per_state[s];

			// start with delta value larger than epsilon such that we go into while loop at least once
			double delta = epsilon + 1;

			while (!(delta < epsilon))
			{

				// Find Max Q value
				// double Q_max = numeric_limits<double>::min();
				double Q_max = -100000;
				for (int a = 0; a < A_max; a++)
				{
					// for (int a = 0; a < A[s].size(); a++){
					if (Q_values_s[a] > Q_max)
					{
						Q_max = Q_values_s[a];
					}
				}

				// best_actions: find those actions that are at most epsilon from largest action
				vector<int> best_actions;
				for (int a = 0; a < A_max; a++)
				{
					// for (int a = 0; a < A[s].size(); a++){
					if (abs(Q_values_s[a] - Q_max) < epsilon)
					{
						best_actions.push_back(a);
					}
				}

				delta = 0.0;

				for (int a : best_actions)
				{
					double old_q = Q_values_s[a];

					// actually update this value Q(s,a)
					auto &[P_s_a, P_s_a_nonzero] = P[s][a];
					Q_values_s[a] = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U_previous_iteration, P_s_a_nonzero);

					if (abs(old_q - Q_values_s[a]) > delta)
					{
						delta = abs(old_q - Q_values_s[a]);
					}
				}
			}

			// find new value of V_U[s]
			// V_U_current_iteration[s] = numeric_limits<double>::min();
			V_U_current_iteration[s] = -100000;
			for (int a = 0; a < A_max; a++)
			{
				// for (int a = 0; a <A[s].size(); a++){
				if (Q_values_s[a] > V_U_current_iteration[s])
				{
					V_U_current_iteration[s] = Q_values_s[a];
				}
			}
		}

		// Check if upper convergence criteria is met
		upper_convergence_criteria = abs_max_diff(V_U[0], V_U[1], S) <= convergence_bound_precomputed;

		// End timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);

		// Record actions eliminated if method is applied
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}

	// Create vector to return
	vector<double> result(S);
	copy(V_U[(iterations & 1)], V_U[(iterations & 1)] + S, result.begin());

	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE THE MEMORY ON THE HEAP
	for (int i = 0; i < 2; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < S; ++i)
	{
		delete[] Q_values_per_state[i];
	}
	delete[] Q_values_per_state;

	// Return the result value
	return result_tuple;
}

V_type value_iteration_BAOSK(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Finds out how big a vector we need to store an action in an entry
	// If the arrays has A_max size, then a = A_max has no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index
	int A_max = find_max_A(A) + 1;

	// Find relevant values from the R parameter
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[2];
	for (int i = 0; i < 2; ++i)
	{
		V_U[i] = new double[S];
	}
	for (int s = 0; s < S; s++)
	{
		V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		V_U[1][s] = 1.0;
	}

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration as a dummy value
	vector<microseconds> work_per_iteration(1);

	// initialize criteria boolean variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged with the given criteria!
	bool upper_convergence_criteria = false;

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// arrays to keep the Q-values
	double **Q_values_per_state = new double *[S];
	for (int i = 0; i < S; ++i)
	{
		Q_values_per_state[i] = new double[A_max];
	}

	//**********************************************************************

	for (int s = 0; s < S; s++)
	{
		// pointers to the heaps of current state s
		double *Q_values_s = Q_values_per_state[s];
		for (int a = 0; a < A_max; a++)
		{
			// Q_values_s[a] = (r_star_max / (1.0 - gamma));	//init with V_max as stated in BAO1 paper
			Q_values_s[a] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s]; // init with upper bound V_U
		}
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;

	// while any of the criteria are NOT, !, met, run the loop
	while (!upper_convergence_criteria)
	{

		// Increment iteration counter
		iterations++;

		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If iiteration is even, then (iteration & 1) is 0, and the one to change is V[0]
		double *V_U_current_iteration = V_U[(iterations & 1)];
		double *V_U_previous_iteration = V_U[1 - (iterations & 1)];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{

			// keep best actions here
			double *Q_values_s = Q_values_per_state[s];

			// start with delta value larger than epsilon such that we go into while loop at least once
			double delta = epsilon + 1;

			while (!(delta < epsilon))
			{

				// Find Max Q value
				double Q_max = numeric_limits<double>::min();
				for (int a = 0; a < A_max; a++)
				{
					if (Q_values_s[a] > Q_max)
					{
						Q_max = Q_values_s[a];
					}
				}

				// best_actions: find those actions that are at most epsilon from largest action
				vector<int> best_actions;
				for (int a = 0; a < A_max; a++)
				{
					if (abs(Q_values_s[a] - Q_max) < epsilon)
					{
						best_actions.push_back(a);
					}
				}

				delta = 0.0;

				for (int a : best_actions)
				{
					double old_q = Q_values_s[a];

					// actually update this value Q(s,a)
					auto &[P_s_a, P_s_a_nonzero] = P[s][a];
					Q_values_s[a] = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U_previous_iteration, P_s_a_nonzero);

					if (abs(old_q - Q_values_s[a]) > delta)
					{
						delta = abs(old_q - Q_values_s[a]);
					}
				}
			}

			// find new value of V_U[s]
			V_U_current_iteration[s] = numeric_limits<double>::min();
			for (int a = 0; a < A_max; a++)
			{
				if (Q_values_s[a] > V_U_current_iteration[s])
				{
					V_U_current_iteration[s] = Q_values_s[a];
				}
			}
		}

		// Check if upper convergence criteria is met
		upper_convergence_criteria = abs_max_diff(V_U[0], V_U[1], S) <= convergence_bound_precomputed;

		// End timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);

		// Record actions eliminated if method is applied
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}

	// Create vector to return
	vector<double> result(S);
	copy(V_U[(iterations & 1)], V_U[(iterations & 1)] + S, result.begin());

	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE THE MEMORY ON THE HEAP
	for (int i = 0; i < 2; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < S; ++i)
	{
		delete[] Q_values_per_state[i];
	}
	delete[] Q_values_per_state;

	// Return the result value
	return result_tuple;
}

V_type value_iteration_BAOSKGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Finds out how big a vector we need to store an action in an entry
	// If the arrays has A_max size, then a = A_max has no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index
	int A_max = find_max_A(A) + 1;

	// Find relevant values from the R parameter
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V_U[i] = new double[S];
	}
	// int siz=sqrt(S-1)-2;
	// int Xmax=siz+2;
	// gamma=1;
	for (int s = 0; s < S; s++)
	{
		/*int x_curr=s%Xmax;
		int y_curr=s/Xmax;
		int xa1= abs(x_curr-siz);
		int ya1= abs(y_curr-siz);
		double x2=0;
		if (xa1>ya1)
			x2=xa1;
		else
			x2=ya1;
		V_U[0][s] =-x2;*/
		V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
	} // V_U[0][S-1] = 0.0;

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration as a dummy value
	vector<microseconds> work_per_iteration(1);

	// initialize criteria boolean variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged with the given criteria!
	bool upper_convergence_criteria = false;

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double convergence_bound_precomputed = 0.0005;
	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// arrays to keep the Q-values
	double **Q_values_per_state = new double *[S];
	for (int i = 0; i < S; ++i)
	{
		// Q_values_per_state[i] = new double[A[i].size()];
		Q_values_per_state[i] = new double[A_max];
	}

	//**********************************************************************

	for (int s = 0; s < S; s++)
	{
		// pointers to the heaps of current state s
		double *Q_values_s = Q_values_per_state[s];
		// for (int a = 0; a < A[s].size(); a++){
		for (int a = 0; a < A_max; a++)
		{
			// Q_values_s[a] = (r_star_max / (1.0 - gamma));	//init with V_max as stated in BAO1 paper
			Q_values_s[a] = V_U[0][s]; //(gamma / (1.0 - gamma)) * r_star_max + r_star_values[s]; //init with upper bound V_U
		}
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;

	// while any of the criteria are NOT, !, met, run the loop
	while (!upper_convergence_criteria)
	{
		upper_convergence_criteria = true;
		// Increment iteration counter
		iterations++;

		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();
		double old_q;
		// If iiteration is even, then (iteration & 1) is 0, and the one to change is V[0]
		double *V_U_current_iteration = V_U[0];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{

			// keep best actions here
			double *Q_values_s = Q_values_per_state[s];

			// start with delta value larger than epsilon such that we go into while loop at least once
			double delta = epsilon + 1;
			// double Q_max = numeric_limits<double>::min();
			double Q_max = -100000;
			// for (int a = 0; a < A[s].size(); a++){
			for (int a = 0; a < A_max; a++)
			{
				if (Q_values_s[a] > Q_max)
				{
					Q_max = Q_values_s[a];
				}
			}
			// vector<int> best_actions2=A[s];
			double oldv = Q_max;
			while (!(delta < epsilon))
			{

				// Find Max Q value

				// best_actions: find those actions that are at most epsilon from largest action

				vector<int> best_actions;
				for (int a = 0; a < A_max; a++)
				{
					// for (int a = 0; a < A[s].size(); a++){
					if (Q_max - Q_values_s[a] < epsilon)
					{
						best_actions.push_back(a);
					}
				}

				delta = 0.0;
				// Q_max=numeric_limits<double>::min();
				Q_max = -100000;
				for (int a : best_actions)
				{
					old_q = Q_values_s[a];

					// actually update this value Q(s,a)
					auto &[P_s_a, P_s_a_nonzero] = P[s][a];
					Q_values_s[a] = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U_current_iteration, P_s_a_nonzero);
					if (abs(old_q - Q_values_s[a]) > delta)
					{
						delta = abs(old_q - Q_values_s[a]);
					}
					// if (Q_values_s[a]>Q_max)
					//	Q_max=Q_values_s[a];
				}
				for (int a = 0; a < A_max; a++)
				{
					// for (int a = 0; a < A[s].size(); a++){
					if (Q_values_s[a] > Q_max)
					{
						Q_max = Q_values_s[a];
					}
					// vector<int> best_actions2=best_actions;
				}

				// find new value of V_U[s]
				// V_U_current_iteration[s] = numeric_limits<double>::min();
			}
			// if (Q_max > V_U_current_iteration[s]) {
			V_U_current_iteration[s] = Q_max;
			//}
			if (abs(oldv - Q_max) > convergence_bound_precomputed)
				upper_convergence_criteria = false;
		}

		// Check if upper convergence criteria is met
		// End timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);

		// Record actions eliminated if method is applied
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}

	// Create vector to return
	vector<double> result(S);
	copy(V_U[0], V_U[0] + S, result.begin());

	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE THE MEMORY ON THE HEAP
	for (int i = 0; i < 1; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < S; ++i)
	{
		delete[] Q_values_per_state[i];
	}
	delete[] Q_values_per_state;

	// Return the result value
	return result_tuple;
}

V_type value_iteration_BAOGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// Finds out how big a vector we need to store an action in an entry
	// If the arrays has A_max size, then a = A_max has no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index

	// int A_max = find_max_A(A) + 1;

	// Find relevant values from the R parameter
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V_U[i] = new double[S];
	}
	// A_maxN[i] = new double[S];
	int A_maxN[S];
	// int siz=sqrt(S-1)-2;
	// int Xmax=siz+2;
	// gamma=1;
	for (int s = 0; s < S; s++)
	{
		/*int x_curr=s%Xmax;
		int y_curr=s/Xmax;
		int xa1= abs(x_curr-siz);
		int ya1= abs(y_curr-siz);
		double x2=0;
		if (xa1>ya1)
			x2=xa1;
		else
			x2=ya1;
			V_U[0][s] =-x2+10;*/
		V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		A_maxN[s] = A[s].size();
	} // V_U[0][S-1] = 0.0;

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration as a dummy value
	vector<microseconds> work_per_iteration(1);

	// initialize criteria boolean variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged with the given criteria!
	bool upper_convergence_criteria = false;

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double convergence_bound_precomputed = 0.0005;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// arrays to keep the Q-values
	double **Q_values_per_state = new double *[S];
	for (int i = 0; i < S; ++i)
	{
		// Q_values_per_state[i] = new double[A[i].size()];
		Q_values_per_state[i] = new double[A_maxN[i]];
	}

	//**********************************************************************

	for (int s = 0; s < S; s++)
	{
		// pointers to the heaps of current state s
		double *Q_values_s = Q_values_per_state[s];
		for (int a = 0; a < A_maxN[s]; a++)
		{
			// Q_values_s[a] = (r_star_max / (1.0 - gamma));	//init with V_max as stated in BAO1 paper
			Q_values_s[a] = V_U[0][s]; //(gamma / (1.0 - gamma)) * r_star_max + r_star_values[s]; //init with upper bound V_U
		}
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;

	// while any of the criteria are NOT, !, met, run the loop
	while (!upper_convergence_criteria)
	{
		upper_convergence_criteria = true;
		// Increment iteration counter
		iterations++;

		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();
		double old_q;
		// If iiteration is even, then (iteration & 1) is 0, and the one to change is V[0]
		double *V_U_current_iteration = V_U[0];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			int sizeO = A_maxN[s];
			// keep best actions here
			double *Q_values_s = Q_values_per_state[s];

			// start with delta value larger than epsilon such that we go into while loop at least once
			double delta = epsilon + 1;
			// double Q_max = numeric_limits<double>::min();
			double Q_max = -100000;

			for (int a = 0; a < sizeO; a++)
			{
				if (Q_values_s[a] > Q_max)
				{
					Q_max = Q_values_s[a];
				}
			}
			// vector<int> best_actions2=A[s];
			double oldv = Q_max;
			while (!(delta < epsilon))
			{

				// Find Max Q value

				// best_actions: find those actions that are at most epsilon from largest action
				vector<int> best_actions;
				for (int a = 0; a < sizeO; a++)
				{
					if (Q_max - Q_values_s[a] < epsilon)
					{
						best_actions.push_back(a);
					}
				}
				delta = 0.0;
				// Q_max=numeric_limits<double>::min();
				Q_max = -100000;
				for (int a : best_actions)
				{
					old_q = Q_values_s[a];

					// actually update this value Q(s,a)
					auto &[P_s_a, P_s_a_nonzero] = P[s][a];
					Q_values_s[a] = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U_current_iteration, P_s_a_nonzero);
					if (abs(old_q - Q_values_s[a]) > delta)
					{
						delta = abs(old_q - Q_values_s[a]);
					}
					// if (Q_values_s[a]>Q_max)
					//	Q_max=Q_values_s[a];
				}

				
				for (int a = 0; a < sizeO; a++)
				{
					if (Q_values_s[a] > Q_max)
					{
						Q_max = Q_values_s[a];
					}
				}
				// vector<int> best_actions2=best_actions;
			}

			// find new value of V_U[s]
			// V_U_current_iteration[s] = numeric_limits<double>::min();
			/*
			for (int a = 0; a < A[s].size(); a++){
					if (Q_values_s[a] > V_U_current_iteration[s]) {
							V_U_current_iteration[s] = Q_values_s[a];
					}
			}/*/
			// if (Q_max > V_U_current_iteration[s]) {
			V_U_current_iteration[s] = Q_max;
			//}
			if (abs(oldv - Q_max) > convergence_bound_precomputed)
				upper_convergence_criteria = false;
		}

		// Check if upper convergence criteria is met
		// End timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);

		// Record actions eliminated if method is applied
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}

	// Create vector to return
	vector<double> result(S);
	copy(V_U[0], V_U[0] + S, result.begin());

	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE THE MEMORY ON THE HEAP
	for (int i = 0; i < 1; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < S; ++i)
	{
		delete[] Q_values_per_state[i];
	}
	delete[] Q_values_per_state;

	// Return the result value
	return result_tuple;
}

V_type value_iteration_BAOGSTM(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon, int D3)
{

	// Finds out how big a vector we need to store an action in an entry
	// If the arrays has A_max size, then a = A_max has no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index

	// int A_max = find_max_A(A) + 1;

	// Find relevant values from the R parameter
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V_U[i] = new double[S];
	}
	// A_maxN[i] = new double[S];
	int A_maxN[S];
	int siz;
	if (D3 == 1)
	{
		siz = sqrt(S - 1) - 2;
	}
	else if (D3 == 2)
	{
		siz = cbrt(S - 1) - 2;
	}
	int Xmax = siz + 2;
	gamma = 1;

	int x_curr;
	int y_curr;
	int z_curr;
	int xa1;
	int ya1;
	int za1;
	double x2;
	if (D3 == 0)
	{
		for (int s = 0; s < S; s++)
		{
			V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
			A_maxN[s] = A[s].size();
		}
	}
	else
	{
		for (int s = 0; s < S; s++)
		{
			if (D3 == 1)
			{
				x_curr = s % Xmax;
				y_curr = s / Xmax;
				xa1 = abs(x_curr - siz);
				ya1 = abs(y_curr - siz);
				za1 = xa1 - 1;
			}
			else
			{
				int idx = s;
				z_curr = idx / (Xmax * Xmax);
				idx -= (z_curr * Xmax * Xmax);
				y_curr = idx / Xmax;
				x_curr = idx % Xmax;
				xa1 = abs(x_curr - siz);
				ya1 = abs(y_curr - siz);
				za1 = abs(z_curr - siz);
			}

			x2 = 0;
			if (xa1 >= ya1 && xa1 >= za1)
				x2 = xa1;
			else if (ya1 >= xa1 && ya1 >= za1)
				x2 = ya1;
			else
				x2 = ya1;
			V_U[0][s] = -x2 + 10;
			V_U[0][s] = 0;
			A_maxN[s] = A[s].size();
		}
		V_U[0][S - 1] = 0.0;
	}
	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration as a dummy value
	vector<microseconds> work_per_iteration(1);

	// initialize criteria boolean variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged with the given criteria!
	bool upper_convergence_criteria = false;

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double convergence_bound_precomputed = 0.0005;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// arrays to keep the Q-values
	double **Q_values_per_state = new double *[S];
	for (int i = 0; i < S; ++i)
	{
		// Q_values_per_state[i] = new double[A[i].size()];
		Q_values_per_state[i] = new double[A_maxN[i]];
	}

	//**********************************************************************

	for (int s = 0; s < S; s++)
	{
		// pointers to the heaps of current state s
		double *Q_values_s = Q_values_per_state[s];
		for (int a = 0; a < A_maxN[s]; a++)
		{
			// Q_values_s[a] = (r_star_max / (1.0 - gamma));	//init with V_max as stated in BAO1 paper
			Q_values_s[a] = V_U[0][s]; //(gamma / (1.0 - gamma)) * r_star_max + r_star_values[s]; //init with upper bound V_U
		}
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;

	// while any of the criteria are NOT, !, met, run the loop
	while (!upper_convergence_criteria)
	{
		upper_convergence_criteria = true;
		// Increment iteration counter
		iterations++;

		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();
		double old_q;
		// If iiteration is even, then (iteration & 1) is 0, and the one to change is V[0]
		double *V_U_current_iteration = V_U[0];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{
			int sizeO = A_maxN[s];
			// keep best actions here
			double *Q_values_s = Q_values_per_state[s];

			// start with delta value larger than epsilon such that we go into while loop at least once
			double delta = epsilon + 1;
			// double Q_max = numeric_limits<double>::min();
			double Q_max = -100000;

			for (int a = 0; a < sizeO; a++)
			{
				if (Q_values_s[a] > Q_max)
				{
					Q_max = Q_values_s[a];
				}
			}
			// vector<int> best_actions2=A[s];
			double oldv = Q_max;
			while (!(delta < epsilon))
			{

				// Find Max Q value

				// best_actions: find those actions that are at most epsilon from largest action
				vector<int> best_actions;
				for (int a = 0; a < sizeO; a++)
				{
					if (Q_max - Q_values_s[a] < epsilon)
					{
						best_actions.push_back(a);
					}
				}
				delta = 0.0;
				// Q_max=numeric_limits<double>::min();
				Q_max = -100000;
				for (int a : best_actions)
				{
					old_q = Q_values_s[a];

					// actually update this value Q(s,a)
					auto &[P_s_a, P_s_a_nonzero] = P[s][a];
					Q_values_s[a] = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U_current_iteration, P_s_a_nonzero);
					if (abs(old_q - Q_values_s[a]) > delta)
					{
						delta = abs(old_q - Q_values_s[a]);
					}
					// if (Q_values_s[a]>Q_max)
					//	Q_max=Q_values_s[a];
				}
				for (int a = 0; a < sizeO; a++)
				{
					if (Q_values_s[a] > Q_max)
					{
						Q_max = Q_values_s[a];
					}
				}
				// vector<int> best_actions2=best_actions;
			}

			// find new value of V_U[s]
			// V_U_current_iteration[s] = numeric_limits<double>::min();
			/*
			for (int a = 0; a < A[s].size(); a++){
					if (Q_values_s[a] > V_U_current_iteration[s]) {
							V_U_current_iteration[s] = Q_values_s[a];
					}
			}/*/
			// if (Q_max > V_U_current_iteration[s]) {
			V_U_current_iteration[s] = Q_max;
			//}
			if (abs(oldv - Q_max) > convergence_bound_precomputed)
				upper_convergence_criteria = false;
		}

		// Check if upper convergence criteria is met
		// End timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);

		// Record actions eliminated if method is applied
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}

	// Create vector to return
	vector<double> result(S);
	copy(V_U[0], V_U[0] + S, result.begin());

	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE THE MEMORY ON THE HEAP
	for (int i = 0; i < 1; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < S; ++i)
	{
		delete[] Q_values_per_state[i];
	}
	delete[] Q_values_per_state;

	// Return the result value
	return result_tuple;
}