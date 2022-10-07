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

// now returns a vector of actions touched in this iteration!
vector<int> heapify_max_actions_touched(int i, q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size)
{

	// init vector to hold actions touched
	vector<int> actions_touched_in_heapify;

	// add the current action
	// actions_touched_in_heapify.push_back(max_heap[i].second);

	int largest = i;

	// the left child index and value
	int l = Left_old(i);

	// The right child index and value
	int r = Right_old(i);

	// if (l < heap_size) {
	//		actions_touched_in_heapify.push_back(max_heap[l].second);
	// }
	if (l < heap_size && max_heap[l].first > max_heap[i].first)
	{
		largest = l;
	}

	// if (r < heap_size) {
	//		actions_touched_in_heapify.push_back(max_heap[r].second);
	// }
	if (r < heap_size && max_heap[r].first > max_heap[largest].first)
	{
		largest = r;
	}

	if (largest != i)
	{
		// record that we bubble down into this action node!
		actions_touched_in_heapify.push_back(max_heap[largest].second);

		q_action_pair_type pair_i = max_heap[i];
		q_action_pair_type pair_largest = max_heap[largest];
		max_heap[i] = pair_largest;
		max_heap[largest] = pair_i;
		max_heap_indicies[pair_i.second] = largest;
		max_heap_indicies[pair_largest.second] = i;
		vector<int> recursive_result = heapify_max_actions_touched(largest, max_heap, max_heap_indicies, heap_size);
		actions_touched_in_heapify.insert(actions_touched_in_heapify.begin(), recursive_result.begin(), recursive_result.end());
	}
	return actions_touched_in_heapify;
}

// now returns a vector of actions touched in this iteration!
vector<int> decrease_max_actions_touched(double newValue, int indexToChange, q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size)
{
	max_heap[indexToChange].first = newValue;
	vector<int> result = heapify_max_actions_touched(indexToChange, max_heap, max_heap_indicies, heap_size);
	result.push_back(max_heap[indexToChange].second);
	return result;
}

V_type value_iteration_actions_touched(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
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

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration
	vector<microseconds> work_per_iteration(1);

	// init criteria variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged!
	bool upper_convergence_criteria = false;

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
	// For testing purposes, fill each entry with -1, so I can see the ones not in the set
	for (int s = 0; s < S; s++)
	{
		fill(heap_max_indicies[s], heap_max_indicies[s] + A_max, -1);
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
	// Heap sizes of each state. Last index of heaps of state s is such heap_size[s] - 1
	// only one variable per state, as the heaps has
	// int heap_size[S];
	int *heap_size = new int[S];

	//**********************************************************************
	// FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN

	for (int s = 0; s < S; s++)
	{

		// pointers to the heaps of current state s
		q_action_pair_type *max_heap_s = max_heaps[s];

		// The initial index in both heaps and the size of the heap
		int initial_index = 0;

		// initialise the heaps with values from the first iteration
		for (int a : A[s])
		{

			// get a pointer to heap indicies arrays of integers
			// a pointer to a array of integers that are the indicies of each action in the heap
			// thus max_ind[a] = index of a in the max heap
			int *max_ind = heap_max_indicies[s];

			// need the distribution to calculate first q-values
			auto &[P_s_a, P_s_a_nonzero] = P[s][a];

			// use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
			double q_1_U_s_a = V_U[0][s]; // R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);
			q_action_pair_type q_a_pair = make_pair(q_1_U_s_a, a);

			// push pair into max heap and record its initial index in the heap array(vector)
			max_heap_s[initial_index] = q_a_pair;

			// increment as last step as to use it as index at first and ends up being the size of the heap
			initial_index = initial_index + 1;
		}

		// set the heap size to correct size
		heap_size[s] = initial_index;

		// MAKE THE ARRAYS BE MAX HEAPS WITH MY OWN HEAP ALGORITHM
		// Make the arrays have the heap property
		build_max_heap(max_heap_s, heap_max_indicies[s], heap_size[s]);
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;

	// while any of the criteria are NOT, !, met, run the loop
	// while NEITHER has converged
	while (!upper_convergence_criteria)
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

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{

			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			V_U_current_iteration[s] = numeric_limits<double>::min();

			// The max heap for the state s
			q_action_pair_type *max_heap_s = max_heaps[s];

			// pointers to the indicies arrays of state s
			int *max_heap_indicies_s = heap_max_indicies[s];

			// START UPDATE update_top_action_max, write in seperate function in the future
			q_action_pair_type top_pair = max_heap_s[0];
			int top_pair_action = top_pair.second;

			auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
			double q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_previous_iteration, P_s_top_pair_nonzero);

			vector<int> actions_touched = decrease_max_actions_touched(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
			for (int a : actions_touched)
			{
				actions_eliminated_in_iteration.push_back(make_pair(s, a));
			}

			// END UPDATE
			int updated_heap_max_top_action = max_heap_s[0].second;

			while (top_pair_action != updated_heap_max_top_action)
			{

				// START UPDATE update_top_action_max, write in seperate function in the future
				q_action_pair_type top_pair = max_heap_s[0];
				top_pair_action = top_pair.second;
				auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
				q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_previous_iteration, P_s_top_pair_nonzero);

				vector<int> actions_touched = decrease_max_actions_touched(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
				for (int a : actions_touched)
				{
					actions_eliminated_in_iteration.push_back(make_pair(s, a));
				}

				// END UPDATE
				updated_heap_max_top_action = max_heap_s[0].second;
			}

			// based on the proven fact that the top action now has the maximum value, that is now set
			V_U_current_iteration[s] = max_heap_s[0].first;
		}

		// see if any of the convergence criteria are met
		// 2. upper criteria
		upper_convergence_criteria = abs_max_diff(V_U[0], V_U[1], S) <= convergence_bound_precomputed;

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}
	// case return value on which convergence criteria was met
	vector<double> result(S); // set it so have size S from beginning to use copy

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
		delete[] max_heaps[i];
	}
	delete[] max_heaps;

	for (int i = 0; i < S; ++i)
	{
		delete[] heap_max_indicies[i];
	}
	delete[] heap_max_indicies;

	delete[] heap_size;

	return result_tuple;
}

V_type value_iteration_actions_touchedGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
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

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration
	vector<microseconds> work_per_iteration(1);

	// init criteria variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged!
	bool upper_convergence_criteria = false;

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
	// For testing purposes, fill each entry with -1, so I can see the ones not in the set
	for (int s = 0; s < S; s++)
	{
		fill(heap_max_indicies[s], heap_max_indicies[s] + A_max, -1);
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
	// Heap sizes of each state. Last index of heaps of state s is such heap_size[s] - 1
	// only one variable per state, as the heaps has
	// int heap_size[S];
	int *heap_size = new int[S];

	//**********************************************************************
	// FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN

	for (int s = 0; s < S; s++)
	{

		// pointers to the heaps of current state s
		q_action_pair_type *max_heap_s = max_heaps[s];

		// The initial index in both heaps and the size of the heap
		int initial_index = 0;

		// initialise the heaps with values from the first iteration
		for (int a : A[s])
		{

			// get a pointer to heap indicies arrays of integers
			// a pointer to a array of integers that are the indicies of each action in the heap
			// thus max_ind[a] = index of a in the max heap
			int *max_ind = heap_max_indicies[s];

			// need the distribution to calculate first q-values
			auto &[P_s_a, P_s_a_nonzero] = P[s][a];

			// use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
			double q_1_U_s_a = V_U[0][s]; // R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);
			q_action_pair_type q_a_pair = make_pair(q_1_U_s_a, a);

			// push pair into max heap and record its initial index in the heap array(vector)
			max_heap_s[initial_index] = q_a_pair;

			// increment as last step as to use it as index at first and ends up being the size of the heap
			initial_index = initial_index + 1;
		}

		// set the heap size to correct size
		heap_size[s] = initial_index;

		// MAKE THE ARRAYS BE MAX HEAPS WITH MY OWN HEAP ALGORITHM
		// Make the arrays have the heap property
		build_max_heap(max_heap_s, heap_max_indicies[s], heap_size[s]);
	}

	//**********************************************************************
	// ACTUAL ITERATIVE VI EFFICIENT ALGORITHM

	// keep count of number of iterations
	int iterations = 0;

	// while any of the criteria are NOT, !, met, run the loop
	// while NEITHER has converged
	double *V_U_current_iteration = V_U[0];
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
		// double *V_U_current_iteration = V_U[(iterations & 1)];
		// double *V_U_previous_iteration = V_U[1 - (iterations & 1)];

		// for all states in each iteration
		for (int s = 0; s < S; s++)
		{

			// TODO if non-negative rewards, 0 is a lower bound of the maximization. to be changed if we want negative rewards
			// V_U_current_iteration[s] = numeric_limits<double>::min();

			// The max heap for the state s
			q_action_pair_type *max_heap_s = max_heaps[s];

			// pointers to the indicies arrays of state s
			int *max_heap_indicies_s = heap_max_indicies[s];

			// START UPDATE update_top_action_max, write in seperate function in the future
			q_action_pair_type top_pair = max_heap_s[0];
			int top_pair_action = top_pair.second;
			double oldv = top_pair.first;

			auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
			double q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_current_iteration, P_s_top_pair_nonzero);

			vector<int> actions_touched = decrease_max_actions_touched(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
			for (int a : actions_touched)
			{
				actions_eliminated_in_iteration.push_back(make_pair(s, a));
			}

			// END UPDATE
			int updated_heap_max_top_action = max_heap_s[0].second;

			while (top_pair_action != updated_heap_max_top_action)
			{

				// START UPDATE update_top_action_max, write in seperate function in the future
				q_action_pair_type top_pair = max_heap_s[0];
				top_pair_action = top_pair.second;
				auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
				q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_current_iteration, P_s_top_pair_nonzero);

				vector<int> actions_touched = decrease_max_actions_touched(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
				for (int a : actions_touched)
				{
					actions_eliminated_in_iteration.push_back(make_pair(s, a));
				}

				// END UPDATE
				updated_heap_max_top_action = max_heap_s[0].second;
			}

			// based on the proven fact that the top action now has the maximum value, that is now set
			V_U_current_iteration[s] = max_heap_s[0].first;
			if (abs(oldv - V_U_current_iteration[s]) > convergence_bound_precomputed)
				upper_convergence_criteria = false;
		}

		// see if any of the convergence criteria are met
		// 2. upper criteria
		// upper_convergence_criteria = abs_max_diff(V_U[0], V_U[1], S) <= convergence_bound_precomputed;

		// end timing of this iteration and record it in work vector
		auto end_of_iteration = high_resolution_clock::now();
		auto duration_of_iteration = duration_cast<microseconds>(end_of_iteration - start_of_iteration);
		work_per_iteration.push_back(duration_of_iteration);
		actions_eliminated.push_back(move(actions_eliminated_in_iteration));
	}
	// case return value on which convergence criteria was met
	vector<double> result(S); // set it so have size S from beginning to use copy

	copy(V_U[0], V_U[0] + S, result.begin());
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// DEALLOCATE THE MEMORY ON THE HEAP
	for (int i = 0; i < 2; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < S; ++i)
	{
		delete[] max_heaps[i];
	}
	delete[] max_heaps;

	for (int i = 0; i < S; ++i)
	{
		delete[] heap_max_indicies[i];
	}
	delete[] heap_max_indicies;

	delete[] heap_size;

	return result_tuple;
}