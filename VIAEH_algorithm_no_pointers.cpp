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

V_type value_iteration_action_elimination_heaps_no_pointers(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// TODO if the arrays has A_max size, then a = A_max hs no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index
	// finds out how big a vector we need to store an action in an entry
	int A_max = find_max_A(A) + 1;

	// Find the maximum reward in the reward table
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

	// 2. Improved Lower Bound
	double **V_L = new double *[2];
	for (int i = 0; i < 2; ++i)
	{
		V_L[i] = new double[S];
	}

	for (int s = 0; s < S; s++)
	{
		V_L[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
		V_L[1][s] = 1.0;
	}

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration
	vector<microseconds> work_per_iteration(1);

	// init criteria variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged!
	bool bounded_convergence_criteria = false;
	bool upper_convergence_criteria = false;
	bool lower_convergence_criteria = false;

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double two_epsilon = 2 * epsilon;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// HEAP INDICIES INITIALIZING
	//  for each s, and each a, gets the index in heap_max and heap_min
	//  init all to 0 with this way in multi dim array to default value which is 0
	//  if A_max is changed, change later code that deppends on it

	// int heap_max_indicies[S][A_max];
	int **heap_max_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		heap_max_indicies[i] = new int[A_max];
	}

	// int heap_min_indicies[S][A_max];
	int **heap_min_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		heap_min_indicies[i] = new int[A_max];
	}

	// For testing purposes, fill each entry with -1, so I can see the ones not in the set
	for (int s = 0; s < S; s++)
	{
		fill(heap_max_indicies[s], heap_max_indicies[s] + A_max, -1);
		fill(heap_min_indicies[s], heap_min_indicies[s] + A_max, -1);
	}

	// HEAP INITIALIZATION
	// TODO: This is the problem that gives a core dump with A_max = 3000
	// TODO inefficient implementation if actions are numbered much higher than there are actual actions.
	// q_action_pair_type max_heaps[S][A_max];
	// q_action_pair_type min_heaps[S][A_max];

	q_action_pair_type **max_heaps = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		max_heaps[i] = new q_action_pair_type[A_max];
	}
	q_action_pair_type **min_heaps = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		min_heaps[i] = new q_action_pair_type[A_max];
	}
	// Heap sizes of each state. Last index of heaps of state s is such heap_size[s] - 1
	// only one variable per state, as the heaps has
	// int heap_size[S];
	int *heap_size = new int[S];

	// vector<heap_of_pairs_type> max_heaps;
	// vector<heap_of_pairs_type> min_heaps;

	//**********************************************************************
	// FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN

	for (int s = 0; s < S; s++)
	{

		// pointers to the heaps of current state s
		q_action_pair_type *max_heap_s = max_heaps[s];
		q_action_pair_type *min_heap_s = min_heaps[s];

		// The initial index in both heaps and the size of the heap
		int initial_index = 0;

		// initialise the heaps with values from the first iteration
		for (int a : A[s])
		{

			// get a pointer to heap indicies arrays of integers
			// a pointer to a array of integers that are the indicies of each action in the heap
			// thus max_ind[a] = index of a in the max heap
			int *max_ind = heap_max_indicies[s];
			int *min_ind = heap_min_indicies[s];

			// need the distribution to calculate first q-values
			auto &[P_s_a, P_s_a_nonzero] = P[s][a];

			// use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
			double q_1_U_s_a = V_U[0][s]; // R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);
			q_action_pair_type q_a_pair = make_pair(q_1_U_s_a, a);

			// push pair into max heap and record its initial index in the heap array(vector)
			max_heap_s[initial_index] = q_a_pair;
			max_ind[a] = initial_index;

			// push pair into min heap and record its initial index in the heap array(vector)
			min_heap_s[initial_index] = q_a_pair;
			min_ind[a] = initial_index;

			// increment as last step as to use it as index at first and ends up being the size of the heap
			initial_index = initial_index + 1;
		}

		// set the heap size to correct size
		heap_size[s] = initial_index;

		// MAKE THE ARRAYS BE MAX HEAPS WITH MY OWN HEAP ALGORITHM
		// Make the arrays have the heap property
		build_max_heap(max_heap_s, heap_max_indicies[s], heap_size[s]);
		build_min_heap(min_heap_s, heap_min_indicies[s], heap_size[s]);
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;

	// while any of the criteria are NOT, !, met, run the loop
	// while NEITHER has converged
	while ((!bounded_convergence_criteria) && (!upper_convergence_criteria) && (!lower_convergence_criteria))
	{

		// Increment iteration counter i
		iterations++;

		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If i is even, then (i & 1) is 0, and the one to change is V[0]
		double *V_U_current_iteration = V_U[(iterations & 1)];
		double *V_U_previous_iteration = V_U[1 - (iterations & 1)];

		double *V_L_current_iteration = V_L[(iterations & 1)];
		double *V_L_previous_iteration = V_L[1 - (iterations & 1)];

		//
		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{

			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			V_U_current_iteration[s] = numeric_limits<double>::min();
			V_L_current_iteration[s] = numeric_limits<double>::min();

			// The max heap for the state s
			q_action_pair_type *max_heap_s = max_heaps[s];
			q_action_pair_type *min_heap_s = min_heaps[s];

			// pointers to the indicies arrays of state s
			int *max_heap_indicies_s = heap_max_indicies[s];
			int *min_heap_indicies_s = heap_min_indicies[s];

			// ranged for loop over all actions in the action set of state s
			// This is changed to use the max_heap as an array of actions in A[s]
			for (int a_index = 0; a_index < heap_size[s]; a_index++)
			{

				// get the action in the a_index of max_heap_s that is used as A[s]
				int a = max_heap_s[a_index].second;

				// reference to the probability vector of size S
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];

				double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_previous_iteration, P_s_a_nonzero);

				if (Q_L_s_a > V_L_current_iteration[s])
				{
					V_L_current_iteration[s] = Q_L_s_a;
				}
			}

			// START UPDATE update_top_action_max, write in seperate function in the future
			q_action_pair_type top_pair = max_heap_s[0];
			int top_pair_action = top_pair.second;

			auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
			double q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_previous_iteration, P_s_top_pair_nonzero);

			decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
			int index_in_min_heap = min_heap_indicies_s[top_pair_action];
			decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

			// END UPDATE
			int updated_heap_max_top_action = max_heap_s[0].second;

			while (top_pair_action != updated_heap_max_top_action)
			{

				// START UPDATE update_top_action_max, write in seperate function in the future
				q_action_pair_type top_pair = max_heap_s[0];
				top_pair_action = top_pair.second;
				auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
				q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_previous_iteration, P_s_top_pair_nonzero);

				decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
				index_in_min_heap = min_heap_indicies_s[top_pair_action];
				decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

				// END UPDATE
				updated_heap_max_top_action = max_heap_s[0].second;
			}

			// based on the proven fact that the top action now has the maximum value, that is now set
			V_U_current_iteration[s] = max_heap_s[0].first;

			// WE NOW START THE ACTION ELIMINATION PROCESS BASED ON THE TWO HEAPS

			// START UPDATE TOP ACTION MIN TODO: make own function

			q_action_pair_type top_pair_min = min_heap_s[0];
			int top_pair_action_min = top_pair_min.second; // was a mistake here that it was top_pair and not top_pair_min

			auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
			double q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero);
			min_heap_s[0].first = q_i_s_a_min;
			int index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
			decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

			// END UPDATE TOP ACTION MIN

			// the first candidate value to be considered
			double updated_min_upper_bound = min_heap_s[0].first;

			while (updated_min_upper_bound < V_L_current_iteration[s])
			{

				// REMOVE THE TOP MIN ACTION
				int removableAction = min_heap_s[0].second;

				// record action eliminated to return
				actions_eliminated_in_iteration.emplace_back(s, removableAction);

				// remove it in the min heap
				remove_index_min_heap(0, min_heap_s, min_heap_indicies_s, heap_size[s]);

				// remove it in the max heap
				int index_in_max_heap = max_heap_indicies_s[removableAction];
				remove_index_max_heap(index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

				// decrease the heap value
				// TODO chosen to do here rather than in either remove method. I have a single heap_size, as they are the same size
				// if both methods decreases, then it is decreased by 2, which is wrong
				heap_size[s]--;

				// START UPDATE TOP ACTION MIN TODO: make own function

				q_action_pair_type top_pair_min = min_heap_s[0];
				top_pair_action_min = top_pair_min.second; // also changed with same mistake
				auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
				q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero);
				min_heap_s[0].first = q_i_s_a_min;
				index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
				decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

				// END UPDATE TOP ACTION MIN
				updated_min_upper_bound = min_heap_s[0].first;
				// TODO commented out below line. Seems I removed top min element twice!
				// remove_index_min_heap(0, min_heap_s, min_heap_indicies_s);
			}
		}

		// see if any of the convergence criteria are met
		// 1. bounded criteria
		bounded_convergence_criteria = abs_max_diff(V_U[(iterations & 1)], V_L[(iterations & 1)], S) <= two_epsilon;

		// 2. upper criteria
		upper_convergence_criteria = abs_max_diff(V_U[0], V_U[1], S) <= convergence_bound_precomputed;

		// 3. lower criteria
		lower_convergence_criteria = abs_max_diff(V_L[0], V_L[1], S) <= convergence_bound_precomputed;

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}
	// case return value on which convergence criteria was met
	vector<double> result(S); // set it so have size S from beginning to use copy

	if (bounded_convergence_criteria)
	{
		result = V_upper_lower_average(V_U[(iterations & 1)], V_L[(iterations & 1)], S);
	}
	else if (upper_convergence_criteria)
	{
		copy(V_U[(iterations & 1)], V_U[(iterations & 1)] + S, result.begin());
	}
	else if (lower_convergence_criteria)
	{
		copy(V_L[(iterations & 1)], V_L[(iterations & 1)] + S, result.begin());
	}
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE THE MEMORY ON THE HEAP
	for (int i = 0; i < 2; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < 2; ++i)
	{
		delete[] V_L[i];
	}
	delete[] V_L;

	for (int i = 0; i < S; ++i)
	{
		delete[] max_heaps[i];
	}
	delete[] max_heaps;

	for (int i = 0; i < S; ++i)
	{
		delete[] min_heaps[i];
	}
	delete[] min_heaps;

	for (int i = 0; i < S; ++i)
	{
		delete[] heap_max_indicies[i];
	}
	delete[] heap_max_indicies;

	for (int i = 0; i < S; ++i)
	{
		delete[] heap_min_indicies[i];
	}
	delete[] heap_min_indicies;

	delete[] heap_size;

	return result_tuple;
}

V_type value_iteration_action_elimination_heaps_no_pointersGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{

	// TODO if the arrays has A_max size, then a = A_max hs no entry as 0 is an action. One fix is to make it 1 bigger to have space for this index
	// finds out how big a vector we need to store an action in an entry
	int A_max = find_max_A(A) + 1;

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[2];
	for (int i = 0; i < 1; ++i)
	{
		V_U[i] = new double[S];
	}
	for (int s = 0; s < S; s++)
	{
		V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
	}

	// 2. Improved Lower Bound
	double **V_L = new double *[2];
	for (int i = 0; i < 1; ++i)
	{
		V_L[i] = new double[S];
	}

	for (int s = 0; s < S; s++)
	{
		V_L[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
	}

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration
	vector<microseconds> work_per_iteration(1);

	// init criteria variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged!
	bool bounded_convergence_criteria = false;
	bool upper_convergence_criteria = false;
	bool lower_convergence_criteria = false;

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double two_epsilon = 2 * epsilon;

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// HEAP INDICIES INITIALIZING
	//  for each s, and each a, gets the index in heap_max and heap_min
	//  init all to 0 with this way in multi dim array to default value which is 0
	//  if A_max is changed, change later code that deppends on it

	// int heap_max_indicies[S][A_max];
	int **heap_max_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		heap_max_indicies[i] = new int[A_max];
	}

	// int heap_min_indicies[S][A_max];
	int **heap_min_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		heap_min_indicies[i] = new int[A_max];
	}

	// For testing purposes, fill each entry with -1, so I can see the ones not in the set
	for (int s = 0; s < S; s++)
	{
		fill(heap_max_indicies[s], heap_max_indicies[s] + A_max, -1);
		fill(heap_min_indicies[s], heap_min_indicies[s] + A_max, -1);
	}

	// HEAP INITIALIZATION
	// TODO: This is the problem that gives a core dump with A_max = 3000
	// TODO inefficient implementation if actions are numbered much higher than there are actual actions.
	// q_action_pair_type max_heaps[S][A_max];
	// q_action_pair_type min_heaps[S][A_max];

	q_action_pair_type **max_heaps = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		max_heaps[i] = new q_action_pair_type[A_max];
	}
	q_action_pair_type **min_heaps = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		min_heaps[i] = new q_action_pair_type[A_max];
	}
	// Heap sizes of each state. Last index of heaps of state s is such heap_size[s] - 1
	// only one variable per state, as the heaps has
	// int heap_size[S];
	int *heap_size = new int[S];

	// vector<heap_of_pairs_type> max_heaps;
	// vector<heap_of_pairs_type> min_heaps;

	//**********************************************************************
	// FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN

	for (int s = 0; s < S; s++)
	{

		// pointers to the heaps of current state s
		q_action_pair_type *max_heap_s = max_heaps[s];
		q_action_pair_type *min_heap_s = min_heaps[s];

		// The initial index in both heaps and the size of the heap
		int initial_index = 0;

		// initialise the heaps with values from the first iteration
		for (int a : A[s])
		{

			// get a pointer to heap indicies arrays of integers
			// a pointer to a array of integers that are the indicies of each action in the heap
			// thus max_ind[a] = index of a in the max heap
			int *max_ind = heap_max_indicies[s];
			int *min_ind = heap_min_indicies[s];

			// need the distribution to calculate first q-values
			auto &[P_s_a, P_s_a_nonzero] = P[s][a];

			// use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
			double q_1_U_s_a = V_U[0][s]; // R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);
			q_action_pair_type q_a_pair = make_pair(q_1_U_s_a, a);

			// push pair into max heap and record its initial index in the heap array(vector)
			max_heap_s[initial_index] = q_a_pair;
			max_ind[a] = initial_index;

			// push pair into min heap and record its initial index in the heap array(vector)
			min_heap_s[initial_index] = q_a_pair;
			min_ind[a] = initial_index;

			// increment as last step as to use it as index at first and ends up being the size of the heap
			initial_index = initial_index + 1;
		}

		// set the heap size to correct size
		heap_size[s] = initial_index;

		// MAKE THE ARRAYS BE MAX HEAPS WITH MY OWN HEAP ALGORITHM
		// Make the arrays have the heap property
		build_max_heap(max_heap_s, heap_max_indicies[s], heap_size[s]);
		build_min_heap(min_heap_s, heap_min_indicies[s], heap_size[s]);
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;

	// while any of the criteria are NOT, !, met, run the loop
	// while NEITHER has converged
	double *V_U_current_iteration = V_U[(iterations & 1)];
	double *V_L_current_iteration = V_L[(iterations & 1)];
	while ((!bounded_convergence_criteria) && (!upper_convergence_criteria) && (!lower_convergence_criteria))
	{
		bounded_convergence_criteria = true;
		upper_convergence_criteria = true;
		lower_convergence_criteria = true;
		// Increment iteration counter i
		iterations++;

		// Record actions eliminated in this iteration over all states
		vector<pair<int, int>> actions_eliminated_in_iteration;

		// begin timing of this iteration
		auto start_of_iteration = high_resolution_clock::now();

		// If i is even, then (i & 1) is 0, and the one to change is V[0]

		//
		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{

			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			// V_U_current_iteration[s] = numeric_limits<double>::min();
			// V_L_current_iteration[s] = numeric_limits<double>::min();
			double oldVU = V_U_current_iteration[s];
			double oldVL = V_L_current_iteration[s];
			// The max heap for the state s
			q_action_pair_type *max_heap_s = max_heaps[s];
			q_action_pair_type *min_heap_s = min_heaps[s];

			// pointers to the indicies arrays of state s
			int *max_heap_indicies_s = heap_max_indicies[s];
			int *min_heap_indicies_s = heap_min_indicies[s];

			// ranged for loop over all actions in the action set of state s
			// This is changed to use the max_heap as an array of actions in A[s]
			for (int a_index = 0; a_index < heap_size[s]; a_index++)
			{

				// get the action in the a_index of max_heap_s that is used as A[s]
				int a = max_heap_s[a_index].second;

				// reference to the probability vector of size S
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];

				double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_current_iteration, P_s_a_nonzero);

				if (Q_L_s_a > V_L_current_iteration[s])
				{
					V_L_current_iteration[s] = Q_L_s_a;
				}
			}

			// START UPDATE update_top_action_max, write in seperate function in the future
			q_action_pair_type top_pair = max_heap_s[0];
			int top_pair_action = top_pair.second;

			auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
			double q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_current_iteration, P_s_top_pair_nonzero);

			decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
			int index_in_min_heap = min_heap_indicies_s[top_pair_action];
			decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

			// END UPDATE
			int updated_heap_max_top_action = max_heap_s[0].second;

			while (top_pair_action != updated_heap_max_top_action)
			{

				// START UPDATE update_top_action_max, write in seperate function in the future
				q_action_pair_type top_pair = max_heap_s[0];
				top_pair_action = top_pair.second;
				auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
				q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_current_iteration, P_s_top_pair_nonzero);

				decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
				index_in_min_heap = min_heap_indicies_s[top_pair_action];
				decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

				// END UPDATE
				updated_heap_max_top_action = max_heap_s[0].second;
			}

			// based on the proven fact that the top action now has the maximum value, that is now set
			V_U_current_iteration[s] = max_heap_s[0].first;

			// WE NOW START THE ACTION ELIMINATION PROCESS BASED ON THE TWO HEAPS

			// START UPDATE TOP ACTION MIN TODO: make own function

			q_action_pair_type top_pair_min = min_heap_s[0];
			int top_pair_action_min = top_pair_min.second; // was a mistake here that it was top_pair and not top_pair_min

			auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
			double q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_current_iteration, P_s_action_min_nonzero);
			min_heap_s[0].first = q_i_s_a_min;
			int index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
			decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

			// END UPDATE TOP ACTION MIN

			// the first candidate value to be considered
			double updated_min_upper_bound = min_heap_s[0].first;

			while (updated_min_upper_bound < V_L_current_iteration[s])
			{

				// REMOVE THE TOP MIN ACTION
				int removableAction = min_heap_s[0].second;

				// record action eliminated to return
				actions_eliminated_in_iteration.emplace_back(s, removableAction);

				// remove it in the min heap
				remove_index_min_heap(0, min_heap_s, min_heap_indicies_s, heap_size[s]);

				// remove it in the max heap
				int index_in_max_heap = max_heap_indicies_s[removableAction];
				remove_index_max_heap(index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

				// decrease the heap value
				// TODO chosen to do here rather than in either remove method. I have a single heap_size, as they are the same size
				// if both methods decreases, then it is decreased by 2, which is wrong
				heap_size[s]--;

				// START UPDATE TOP ACTION MIN TODO: make own function

				q_action_pair_type top_pair_min = min_heap_s[0];
				top_pair_action_min = top_pair_min.second; // also changed with same mistake
				auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
				q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_current_iteration, P_s_action_min_nonzero);
				min_heap_s[0].first = q_i_s_a_min;
				index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
				decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

				// END UPDATE TOP ACTION MIN
				updated_min_upper_bound = min_heap_s[0].first;
				// TODO commented out below line. Seems I removed top min element twice!
				// remove_index_min_heap(0, min_heap_s, min_heap_indicies_s);
			}
			if ((V_U_current_iteration[s] - V_L_current_iteration[s]) > two_epsilon)
				bounded_convergence_criteria = false;
			if (abs(V_U_current_iteration[s] - oldVU) > convergence_bound_precomputed)
				upper_convergence_criteria = false;
			if (abs(V_L_current_iteration[s] - oldVL) > convergence_bound_precomputed)
				lower_convergence_criteria = false;
		}

		// see if any of the convergence criteria are met
		// 1. bounded criteria
		// bounded_convergence_criteria = abs_max_diff(V_U[(iterations & 1)], V_L[(iterations & 1)], S) <= two_epsilon;

		// 2. upper criteria
		// upper_convergence_criteria = abs_max_diff(V_U[0], V_U[1], S) <= convergence_bound_precomputed;

		// 3. lower criteria
		// lower_convergence_criteria = abs_max_diff(V_L[0], V_L[1], S) <= convergence_bound_precomputed;

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}
	// case return value on which convergence criteria was met
	vector<double> result(S); // set it so have size S from beginning to use copy

	if (bounded_convergence_criteria)
	{
		result = V_upper_lower_average(V_U[(0)], V_L[(0)], S);
	}
	else if (upper_convergence_criteria)
	{
		copy(V_U[0], V_U[(0)] + S, result.begin());
	}
	else if (lower_convergence_criteria)
	{
		copy(V_L[(0)], V_L[(0)] + S, result.begin());
	}
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE THE MEMORY ON THE HEAP
	for (int i = 0; i < 2; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < 2; ++i)
	{
		delete[] V_L[i];
	}
	delete[] V_L;

	for (int i = 0; i < S; ++i)
	{
		delete[] max_heaps[i];
	}
	delete[] max_heaps;

	for (int i = 0; i < S; ++i)
	{
		delete[] min_heaps[i];
	}
	delete[] min_heaps;

	for (int i = 0; i < S; ++i)
	{
		delete[] heap_max_indicies[i];
	}
	delete[] heap_max_indicies;

	for (int i = 0; i < S; ++i)
	{
		delete[] heap_min_indicies[i];
	}
	delete[] heap_min_indicies;

	delete[] heap_size;

	return result_tuple;
}
