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

using namespace std;
using namespace std::chrono;

int Parent(int i)
{
	return (i - 1) / 2;
}

int Left(int i)
{
	return 2 * i + 1;
}

int Right(int i)
{
	return 2 * i + 2;
}

void heapify_max(int i, q_action_pair_type *max_heap[], int max_heap_indicies[], int heap_size)
{

	int largest = i;

	// the left child index and value
	int l = Left(i);

	// The right child index and value
	int r = Right(i);

	if (l < heap_size && (*max_heap[l]).first > (*max_heap[i]).first)
	{
		largest = l;
	}

	if (r < heap_size && (*max_heap[r]).first > (*max_heap[largest]).first)
	{
		largest = r;
	}

	if (largest != i)
	{

		// try these if swap does not work
		// q_action_pair_type *pair_i = max_heap[i];
		// q_action_pair_type *pair_largest = max_heap[largest];
		// max_heap[i] = pair_largest;
		// max_heap[largest] = pair_i;
		// max_heap_indicies[(*pair_i).second] = largest;
		// max_heap_indicies[(*pair_largest).second] = i;

		// use std::swap for efficiency
		swap(max_heap[i], max_heap[largest]);
		swap(max_heap_indicies[(*max_heap[i]).second], max_heap_indicies[(*max_heap[largest]).second]);
		heapify_max(largest, max_heap, max_heap_indicies, heap_size);
	}
}

void heapify_min(int i, q_action_pair_type *min_heap[], int min_heap_indicies[], int heap_size)
{

	int smallest = i;

	// the left child index and value
	int l = Left(i);

	// The right child index and value
	int r = Right(i);

	if (l < heap_size && (*min_heap[l]).first < (*min_heap[i]).first)
	{
		smallest = l;
	}

	if (r < heap_size && (*min_heap[r]).first < (*min_heap[smallest]).first)
	{
		smallest = r;
	}

	if (smallest != i)
	{
		// q_action_pair_type *pair_i = min_heap[i];
		// q_action_pair_type *pair_smallest = min_heap[smallest];
		// min_heap[i] = pair_smallest;
		// min_heap[smallest] = pair_i;
		// min_heap_indicies[(*pair_i).second] = smallest;
		// min_heap_indicies[(*pair_smallest).second] = i;

		// use std::swap
		swap(min_heap[i], min_heap[smallest]);
		swap(min_heap_indicies[(*min_heap[i]).second], min_heap_indicies[(*min_heap[smallest]).second]);
		heapify_min(smallest, min_heap, min_heap_indicies, heap_size);
	}
}

void decrease_max(double newValue, int indexToChange, q_action_pair_type *max_heap[], int max_heap_indicies[], int heap_size)
{
	(*max_heap[indexToChange]).first = newValue;
	heapify_max(indexToChange, max_heap, max_heap_indicies, heap_size);
}

// does not need heap size as the index decreases
void decrease_min(double newValue, int indexToChange, q_action_pair_type *min_heap[], int min_heap_indicies[])
{
	(*min_heap[indexToChange]).first = newValue;
	int i = indexToChange;
	while (i > 0 && (*min_heap[Parent(i)]).first > (*min_heap[i]).first)
	{

		// try this if swap does not work
		// q_action_pair_type *pair_i = min_heap[i];
		// q_action_pair_type *pair_parent = min_heap[Parent(i)];
		// min_heap[i] = pair_parent;
		// min_heap[Parent(i)] = pair_i;
		// min_heap_indicies[(*pair_i).second] = Parent(i);
		// min_heap_indicies[(*pair_parent).second] = i;

		// swap pointers
		swap(min_heap[i], min_heap[Parent(i)]);
		swap(min_heap_indicies[(*min_heap[i]).second], min_heap_indicies[(*min_heap[Parent(i)]).second]);
		i = Parent(i);
	}
}

// TODO: what if the heap is empty, will not happen in our case, as we cannot with < prune away last action. Safeguard?
// TODO: the heap size is decreased before this is called. Could be included in the method instead
void remove_index_max_heap(int indexToRemove, q_action_pair_type *max_heap[], int max_heap_indicies[], int heap_size)
{

	// The last index of the array/vector that holds the heap
	// after the function is called, this index is not included in the heap anymore
	// as the heap size is decreased by one
	int lastElementIndex = heap_size - 1;

	// record the action changed for recording of the heap
	int action_last_element = (*max_heap[lastElementIndex]).second;

	// This can be removed, if not relveant for viewing
	int action_to_remove = (*max_heap[indexToRemove]).second;

	// Set the index to remove to the lastElementIndex
	max_heap[indexToRemove] = max_heap[lastElementIndex];

	// record the change of index in the max_heap_index
	max_heap_indicies[action_last_element] = indexToRemove;

	// OBS: this is moved below the one where index of last element is set
	// became a problem if the last element is the one to remove. Then it was set to -1 and
	// then to the index of last element which was wrong, at it was removed
	// record that the action is now not in A[s] by setting to -1
	max_heap_indicies[action_to_remove] = -1;

	// now heapify the heap from index indexToRemove, which now breaks the heap-property
	heapify_max(indexToRemove, max_heap, max_heap_indicies, heap_size - 1);
}

// what if the heap is empty, will not happen in our case, as we cannot with < prune away last action. Safeguard?
void remove_index_min_heap(int indexToRemove, q_action_pair_type *min_heap[], int min_heap_indicies[], int heap_size)
{

	// The last index of the array/vector that holds the heap
	int lastElementIndex = heap_size - 1;

	// record the action changed for recording of the heap
	int action_last_element = (*min_heap[lastElementIndex]).second;

	// record that the action is now not in A[s] by setting to -1
	// This can be removed, if not relveant for viewing
	int action_to_remove = (*min_heap[indexToRemove]).second;
	min_heap_indicies[action_to_remove] = -1;

	// Set the index to remove to the lastElementIndex
	min_heap[indexToRemove] = min_heap[lastElementIndex];

	// record the change of index in the min_heap_index
	min_heap_indicies[action_last_element] = indexToRemove;

	// now heapify the heap from index indexToRemove, which now breaks the heap-property
	heapify_min(indexToRemove, min_heap, min_heap_indicies, heap_size - 1);
}

void build_max_heap(q_action_pair_type *max_heap[], int max_heap_indicies[], int heap_size)
{

	// This has implicit floor in the division, which is the correct way
	int index_of_last_non_leaf_node = (heap_size / 2) - 1;

	for (int i = index_of_last_non_leaf_node; i >= 0; i--)
	{
		heapify_max(i, max_heap, max_heap_indicies, heap_size);
	}
}

void build_min_heap(q_action_pair_type *min_heap[], int min_heap_indicies[], int heap_size)
{

	// This has implicit floor in the division, which is the correct way
	int index_of_last_non_leaf_node = (heap_size / 2) - 1;

	for (int i = index_of_last_non_leaf_node; i >= 0; i--)
	{
		heapify_min(i, min_heap, min_heap_indicies, heap_size);
	}
}

V_type value_iteration_action_elimination_heaps(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{
	int A_max = find_max_A(A) + 1;

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[2];
	double **V_L = new double *[2];
	for (int i = 0; i < 2; ++i)
	{
		V_U[i] = new double[S];
		V_L[i] = new double[S];
	}
	int siz = sqrt(S - 1) - 2;
	int Xmax = siz + 2;
	// siz=Xmax/2;
	for (int s = 0; s < S; s++)
	{
		// V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		V_U[1][s] = 1;
		int x_curr = s % Xmax;
		int y_curr = s / Xmax;
		double x1 = sqrt(pow(abs(x_curr - siz), 2) + pow(abs(y_curr - siz), 2));
		int xa1 = abs(x_curr - siz);
		int ya1 = abs(y_curr - siz);
		double x2 = 0;
		if (xa1 > ya1)
			x2 = xa1;
		else
			x2 = ya1;
		// double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
		V_U[0][s] = -x2;
		V_L[0][s] = -x1 * 5 - 10;
		V_U[1][s] = 1;
		V_L[1][s] = 1.0;
	}
	V_U[0][S - 1] = 0;
	V_L[0][S - 1] = 0;

	// 2. Improved Lower Bound

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration
	vector<microseconds> work_per_iteration(1);

	// init criteria variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged!
	bool bounded_convergence_criteria = false;
	bool upper_convergence_criteria = false;
	bool lower_convergence_criteria = false;

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double convergence_bound_precomputed = 0.0005;
	const double two_epsilon = 2 * epsilon;

	// HEAP INDICIES INITIALIZING
	//  for each s, and each a, gets the index in heap_max and heap_min
	//  init all to 0 with this way in multi dim array to default value which is 0
	//  if A_max is changed, change later code that deppends on it

	// int heap_max_indicies[S][A_max];
	int **heap_max_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		// heap_max_indicies[i] = new int[A_max];
		heap_max_indicies[i] = new int[A[i].size()];
	}

	// int heap_min_indicies[S][A_max];
	int **heap_min_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		// heap_min_indicies[i] = new int[A_max];
		heap_min_indicies[i] = new int[A[i].size()];
	}

	// For testing purposes, fill each entry with -1, so I can see the ones not in the set
	for (int s = 0; s < S; s++)
	{
		// fill(heap_max_indicies[s], heap_max_indicies[s] + A_max, -1);
		// fill(heap_min_indicies[s], heap_min_indicies[s] + A_max, -1);
		fill(heap_max_indicies[s], heap_max_indicies[s] + A[s].size(), 3);
		fill(heap_min_indicies[s], heap_min_indicies[s] + A[s].size(), 3);
	}

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());
	// HEAP INITIALIZATION

	// q_action_pair_type *max_heaps[S][A_max];
	q_action_pair_type ***max_heaps = new q_action_pair_type **[S];
	for (int i = 0; i < S; ++i)
	{
		max_heaps[i] = new q_action_pair_type *[A[i].size()];
	}

	// q_action_pair_type *min_heaps[S][A_max];
	q_action_pair_type ***min_heaps = new q_action_pair_type **[S];
	for (int i = 0; i < S; ++i)
	{
		min_heaps[i] = new q_action_pair_type *[A[i].size()];
	}

	// The underlying arrays, for each state, of the pairs
	// holds the actual objects, but they never move

	// q_action_pair_type max_heaps_object_arrays[S][A_max];
	q_action_pair_type **max_heaps_object_arrays = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		max_heaps_object_arrays[i] = new q_action_pair_type[A[i].size()];
	}

	// q_action_pair_type min_heaps_object_arrays[S][A_max];
	q_action_pair_type **min_heaps_object_arrays = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		min_heaps_object_arrays[i] = new q_action_pair_type[A[i].size()];
	}

	// Heap sizes of each state. Last index of heaps of state s is such heap_size[s] - 1
	// only one variable per state, as the heaps has
	int *heap_size = new int[S];

	//**********************************************************************
	// FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN
	for (int s = 0; s < S; s++)
	{

		// pointers to the heaps of current state s
		// remember, an array varible is just a pointer to the beginning of the array
		// now, it is an pointer to an array
		q_action_pair_type **max_heap_s = max_heaps[s];
		q_action_pair_type **min_heap_s = min_heaps[s];

		// The underlying array with the actual objects
		// place the objects here
		q_action_pair_type *max_heap_s_object_array = max_heaps_object_arrays[s];
		q_action_pair_type *min_heap_s_object_array = min_heaps_object_arrays[s];

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
			double q_1_U_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);
			q_action_pair_type q_a_pair = make_pair(q_1_U_s_a, a);

			// push pair into max heap and record its initial index in the heap array(vector)
			// seems most natural to put the pair of action a at index a
			max_heap_s_object_array[a] = q_a_pair;
			max_heap_s[initial_index] = &max_heap_s_object_array[a];
			max_ind[a] = initial_index;

			// push pair into min heap and record its initial index in the heap array(vector)
			min_heap_s_object_array[a] = q_a_pair;
			min_heap_s[initial_index] = &min_heap_s_object_array[a];
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
	gamma = 1;
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
			// V_U_current_iteration[s] = numeric_limits<double>::min();
			// V_L_current_iteration[s] = numeric_limits<double>::min();
			V_U_current_iteration[s] = -100000;
			V_L_current_iteration[s] = -100000;

			// The max heap for the state s
			q_action_pair_type **max_heap_s = max_heaps[s];
			q_action_pair_type **min_heap_s = min_heaps[s];

			// pointers to the indicies arrays of state s
			int *max_heap_indicies_s = heap_max_indicies[s];
			int *min_heap_indicies_s = heap_min_indicies[s];

			// keep the rewards from the upper bound for
			int A_s_size = heap_size[s];
			double Q_U_s[A_s_size];

			// ranged for loop over all actions in the action set of state s
			// This is changed to use the max_heap as an array of actions in A[s]
			for (int a_index = 0; a_index < A_s_size; a_index++)
			{

				// get the action in the a_index of max_heap_s that is used as A[s]
				int a = (*max_heap_s[a_index]).second;

				// reference to the probability vector of size S
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];

				double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_previous_iteration, P_s_a_nonzero);

				if (Q_L_s_a > V_L_current_iteration[s])
				{
					V_L_current_iteration[s] = Q_L_s_a;
				}
			}

			// Beginning of updating top upper bound actions
			int heap_max_top_action = (*max_heap_s[0]).second;

			// START UPDATE update_top_action_max, write in seperate function in the future
			q_action_pair_type top_pair = (*max_heap_s[0]);
			int top_pair_action = top_pair.second;

			auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
			double q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_previous_iteration, P_s_top_pair_nonzero);

			decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
			int index_in_min_heap = min_heap_indicies_s[top_pair_action];
			decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

			// END UPDATE
			int updated_heap_max_top_action = (*max_heap_s[0]).second;

			while (top_pair_action != updated_heap_max_top_action)
			{
				// Beginning of updating top upper bound actions
				heap_max_top_action = (*max_heap_s[0]).second;

				// START UPDATE update_top_action_max, write in seperate function in the future
				q_action_pair_type top_pair = *max_heap_s[0];
				top_pair_action = top_pair.second;
				auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
				q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_previous_iteration, P_s_top_pair_nonzero);

				decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
				index_in_min_heap = min_heap_indicies_s[top_pair_action];
				decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

				// END UPDATE
				updated_heap_max_top_action = (*max_heap_s[0]).second;
			}

			// based on the proven fact that the top action now has the maximum value, that is now set
			V_U_current_iteration[s] = (*max_heap_s[0]).first;

			// WE NOW START THE ACTION ELIMINATION PROCESS BASED ON THE TWO HEAPS

			// START UPDATE TOP ACTION MIN TODO: make own function

			q_action_pair_type top_pair_min = *min_heap_s[0];
			int top_pair_action_min = top_pair_min.second; // was a mistake here that it was top_pair and not top_pair_min

			auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
			double q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero);
			(*min_heap_s[0]).first = q_i_s_a_min;
			int index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
			decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

			// END UPDATE TOP ACTION MIN

			// the first candidate value to be considered
			double updated_min_upper_bound = (*min_heap_s[0]).first;

			while (updated_min_upper_bound < V_L_current_iteration[s])
			{

				// REMOVE THE TOP MIN ACTION
				int removableAction = (*min_heap_s[0]).second;

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

				q_action_pair_type top_pair_min = (*min_heap_s[0]);
				top_pair_action_min = top_pair_min.second; // also changed with same mistake
				auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
				q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero);
				(*min_heap_s[0]).first = q_i_s_a_min;
				index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
				decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

				// END UPDATE TOP ACTION MIN
				updated_min_upper_bound = (*min_heap_s[0]).first;
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

	// deallocate the memory
	delete[] heap_size;

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
		delete[] max_heaps_object_arrays[i];
	}
	delete[] max_heaps_object_arrays;

	for (int i = 0; i < S; ++i)
	{
		delete[] min_heaps_object_arrays[i];
	}
	delete[] min_heaps_object_arrays;

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

	return result_tuple;
}

V_type value_iteration_action_elimination_heapsGS(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
{
	int A_max = find_max_A(A) + 1;

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[1];
	double **V_L = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V_U[i] = new double[S];
		V_L[i] = new double[S];
	}
	// int siz=sqrt(S-1)-2;
	// int Xmax=siz+2;
	// gamma=1;
	for (int s = 0; s < S; s++)
	{
		V_U[0][s] = (gamma / (1.0 - gamma)) * r_star_max + r_star_values[s];
		V_L[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
		/*int x_curr=s%Xmax;
			int y_curr=s/Xmax;
			double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
			int xa1= abs(x_curr-siz);
			int ya1= abs(y_curr-siz);
			double x2=0;
			if (xa1>ya1)
				x2=xa1;
			else
				x2=ya1;
			//double x1= sqrt( pow( abs(x_curr-siz),2)+pow(abs(y_curr-siz),2));
			V_U[0][s] = -x2+10;
			V_L[0][s] = -x1*5-10;*/
	} // V_U[0][S-1] = 0;
	// V_L[0][S-1] = 0;

	// 2. Improved Lower Bound

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration
	vector<microseconds> work_per_iteration(1);

	// init criteria variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged!
	bool bounded_convergence_criteria = false;
	bool upper_convergence_criteria = false;
	bool lower_convergence_criteria = false;

	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double two_epsilon = 2 * epsilon;
	const double convergence_bound_precomputed = 0.0005;

	// HEAP INDICIES INITIALIZING
	//  for each s, and each a, gets the index in heap_max and heap_min
	//  init all to 0 with this way in multi dim array to default value which is 0
	//  if A_max is changed, change later code that deppends on it

	// int heap_max_indicies[S][A_max];
	int **heap_max_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		// heap_max_indicies[i] = new int[A_max];
		heap_max_indicies[i] = new int[A[i].size()];
	}

	// int heap_min_indicies[S][A_max];
	int **heap_min_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		// heap_min_indicies[i] = new int[A_max];
		heap_min_indicies[i] = new int[A[i].size()];
	}

	// For testing purposes, fill each entry with -1, so I can see the ones not in the set
	for (int s = 0; s < S; s++)
	{
		// fill(heap_max_indicies[s], heap_max_indicies[s] + A_max, -1);
		// fill(heap_min_indicies[s], heap_min_indicies[s] + A_max, -1);
		fill(heap_max_indicies[s], heap_max_indicies[s] + A[s].size(), -1);
		fill(heap_min_indicies[s], heap_min_indicies[s] + A[s].size(), -1);
	}

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// HEAP INITIALIZATION

	// q_action_pair_type *max_heaps[S][A_max];
	q_action_pair_type ***max_heaps = new q_action_pair_type **[S];
	for (int i = 0; i < S; ++i)
	{
		// max_heaps[i] = new q_action_pair_type*[A_max];
		max_heaps[i] = new q_action_pair_type *[A[i].size()];
	}

	// q_action_pair_type *min_heaps[S][A_max];
	q_action_pair_type ***min_heaps = new q_action_pair_type **[S];
	for (int i = 0; i < S; ++i)
	{
		// min_heaps[i] = new q_action_pair_type*[A_max];
		min_heaps[i] = new q_action_pair_type *[A[i].size()];
	}

	// The underlying arrays, for each state, of the pairs
	// holds the actual objects, but they never move

	// q_action_pair_type max_heaps_object_arrays[S][A_max];
	q_action_pair_type **max_heaps_object_arrays = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		// max_heaps_object_arrays[i] = new q_action_pair_type[A_max];
		max_heaps_object_arrays[i] = new q_action_pair_type[A[i].size()];
	}

	// q_action_pair_type min_heaps_object_arrays[S][A_max];
	q_action_pair_type **min_heaps_object_arrays = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		// min_heaps_object_arrays[i] = new q_action_pair_type[A_max];
		min_heaps_object_arrays[i] = new q_action_pair_type[A[i].size()];
	}

	// Heap sizes of each state. Last index of heaps of state s is such heap_size[s] - 1
	// only one variable per state, as the heaps has
	int *heap_size = new int[S];

	//**********************************************************************
	// FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN

	for (int s = 0; s < S; s++)
	{

		// pointers to the heaps of current state s
		// remember, an array varible is just a pointer to the beginning of the array
		// now, it is an pointer to an array
		q_action_pair_type **max_heap_s = max_heaps[s];
		q_action_pair_type **min_heap_s = min_heaps[s];

		// The underlying array with the actual objects
		// place the objects here
		q_action_pair_type *max_heap_s_object_array = max_heaps_object_arrays[s];
		q_action_pair_type *min_heap_s_object_array = min_heaps_object_arrays[s];

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
			double q_1_U_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);
			q_action_pair_type q_a_pair = make_pair(q_1_U_s_a, a);

			// push pair into max heap and record its initial index in the heap array(vector)
			// seems most natural to put the pair of action a at index a
			max_heap_s_object_array[a] = q_a_pair;
			max_heap_s[initial_index] = &max_heap_s_object_array[a];
			max_ind[a] = initial_index;

			// push pair into min heap and record its initial index in the heap array(vector)
			min_heap_s_object_array[a] = q_a_pair;
			min_heap_s[initial_index] = &min_heap_s_object_array[a];
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
	double *V_U_current_iteration = V_U[0];
	double *V_L_current_iteration = V_L[0];
	while ((!bounded_convergence_criteria) && (!upper_convergence_criteria) && (!lower_convergence_criteria))
	{
		bounded_convergence_criteria = true;
		upper_convergence_criteria = true;
		lower_convergence_criteria = true;
		// Increment iteration counter i
		iterations++;

		// Record actions eliminated in this iteration over all states
		//AFRT
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
			q_action_pair_type **max_heap_s = max_heaps[s];
			q_action_pair_type **min_heap_s = min_heaps[s];

			// pointers to the indicies arrays of state s
			int *max_heap_indicies_s = heap_max_indicies[s];
			int *min_heap_indicies_s = heap_min_indicies[s];
			double Q_max = numeric_limits<double>::min();
			// keep the rewards from the upper bound for
			int A_s_size = heap_size[s];
			double Q_U_s[A_s_size];

			// ranged for loop over all actions in the action set of state s
			// This is changed to use the max_heap as an array of actions in A[s]
			for (int a_index = 0; a_index < A_s_size; a_index++)
			{

				// get the action in the a_index of max_heap_s that is used as A[s]
				int a = (*max_heap_s[a_index]).second;

				// reference to the probability vector of size S
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];

				double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_current_iteration, P_s_a_nonzero);

				if (Q_L_s_a > V_L_current_iteration[s])
				{
					V_L_current_iteration[s] = Q_L_s_a;
				}
			}

			// Beginning of updating top upper bound actions
			int heap_max_top_action = (*max_heap_s[0]).second;

			// START UPDATE update_top_action_max, write in seperate function in the future
			q_action_pair_type top_pair = (*max_heap_s[0]);
			int top_pair_action = top_pair.second;

			auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
			double q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_current_iteration, P_s_top_pair_nonzero);

			decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
			int index_in_min_heap = min_heap_indicies_s[top_pair_action];
			decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

			// END UPDATE
			int updated_heap_max_top_action = (*max_heap_s[0]).second;

			while (top_pair_action != updated_heap_max_top_action)
			{
				// Beginning of updating top upper bound actions
				heap_max_top_action = (*max_heap_s[0]).second;

				// START UPDATE update_top_action_max, write in seperate function in the future
				q_action_pair_type top_pair = *max_heap_s[0];
				top_pair_action = top_pair.second;
				auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
				q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_current_iteration, P_s_top_pair_nonzero);

				decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
				index_in_min_heap = min_heap_indicies_s[top_pair_action];
				decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

				// END UPDATE
				updated_heap_max_top_action = (*max_heap_s[0]).second;
			}

			// based on the proven fact that the top action now has the maximum value, that is now set
			V_U_current_iteration[s] = (*max_heap_s[0]).first;

			// WE NOW START THE ACTION ELIMINATION PROCESS BASED ON THE TWO HEAPS

			// START UPDATE TOP ACTION MIN TODO: make own function

			q_action_pair_type top_pair_min = *min_heap_s[0];
			int top_pair_action_min = top_pair_min.second; // was a mistake here that it was top_pair and not top_pair_min

			auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
			double q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_current_iteration, P_s_action_min_nonzero);
			(*min_heap_s[0]).first = q_i_s_a_min;
			int index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
			decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

			// END UPDATE TOP ACTION MIN

			// the first candidate value to be considered
			double updated_min_upper_bound = (*min_heap_s[0]).first;

			while (updated_min_upper_bound < V_L_current_iteration[s])
			{

				// REMOVE THE TOP MIN ACTION
				int removableAction = (*min_heap_s[0]).second;

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

				q_action_pair_type top_pair_min = (*min_heap_s[0]);
				top_pair_action_min = top_pair_min.second; // also changed with same mistake
				auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
				q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_current_iteration, P_s_action_min_nonzero);
				(*min_heap_s[0]).first = q_i_s_a_min;
				index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
				decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

				// END UPDATE TOP ACTION MIN
				updated_min_upper_bound = (*min_heap_s[0]).first;
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
		copy(V_U[(0)], V_U[(0)] + S, result.begin());
	}
	else if (lower_convergence_criteria)
	{
		copy(V_L[(0)], V_L[(0)] + S, result.begin());
	}
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// deallocate the memory
	delete[] heap_size;

	for (int i = 0; i < 1; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < 1; ++i)
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
		delete[] max_heaps_object_arrays[i];
	}
	delete[] max_heaps_object_arrays;

	for (int i = 0; i < S; ++i)
	{
		delete[] min_heaps_object_arrays[i];
	}
	delete[] min_heaps_object_arrays;

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

	return result_tuple;
}

V_type value_iteration_action_elimination_heapsGSTM(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon, int D3)
{
	int A_max = find_max_A(A) + 1;

	// Find the maximum reward in the reward table
	auto [r_star_min, r_star_max, r_star_values] = find_all_r_values(R);

	// 1. Improved Upper Bound
	double **V_U = new double *[1];
	double **V_L = new double *[1];
	for (int i = 0; i < 1; ++i)
	{
		V_U[i] = new double[S];
		V_L[i] = new double[S];
	}
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
			V_L[0][s] = (gamma / (1.0 - gamma)) * r_star_min + r_star_values[s];
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
			V_L[0][s] = -500;
		}
		V_U[0][S - 1] = 0.0;
		V_L[0][S - 1] = 0;
	}

	// 2. Improved Lower Bound

	// keep track of work done in each iteration in microseconds
	// start from iteration 1, so put a 0 value into the first iteration
	vector<microseconds> work_per_iteration(1);

	// init criteria variables to know which value to return based on why the algorithm terminated
	// set to true if we have converged!
	bool bounded_convergence_criteria = false;
	bool upper_convergence_criteria = false;
	bool lower_convergence_criteria = false;
	// pre-compute convergence criteria for efficiency to not do it in each iteration of while loop
	// const double convergence_bound_precomputed = (epsilon * (1.0 - gamma)) / gamma;
	const double two_epsilon = 2 * epsilon;
	const double convergence_bound_precomputed = 0.0005;

	// HEAP INDICIES INITIALIZING
	//  for each s, and each a, gets the index in heap_max and heap_min
	//  init all to 0 with this way in multi dim array to default value which is 0
	//  if A_max is changed, change later code that deppends on it

	// int heap_max_indicies[S][A_max];
	int **heap_max_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		// heap_max_indicies[i] = new int[A_max];
		heap_max_indicies[i] = new int[A[i].size()];
	}

	// int heap_min_indicies[S][A_max];
	int **heap_min_indicies = new int *[S];
	for (int i = 0; i < S; ++i)
	{
		// heap_min_indicies[i] = new int[A_max];
		heap_min_indicies[i] = new int[A[i].size()];
	}

	// For testing purposes, fill each entry with -1, so I can see the ones not in the set
	for (int s = 0; s < S; s++)
	{
		// fill(heap_max_indicies[s], heap_max_indicies[s] + A_max, -1);
		// fill(heap_min_indicies[s], heap_min_indicies[s] + A_max, -1);
		fill(heap_max_indicies[s], heap_max_indicies[s] + A[s].size(), -1);
		fill(heap_min_indicies[s], heap_min_indicies[s] + A[s].size(), -1);
	}

	// record actions eliminated in each iteration, where a pair is (state, action)
	// push empty vector for 0-index. Iterations start with 1
	vector<vector<pair<int, int>>> actions_eliminated;
	actions_eliminated.push_back(vector<pair<int, int>>());

	// HEAP INITIALIZATION

	// q_action_pair_type *max_heaps[S][A_max];
	q_action_pair_type ***max_heaps = new q_action_pair_type **[S];
	for (int i = 0; i < S; ++i)
	{
		// max_heaps[i] = new q_action_pair_type*[A_max];
		max_heaps[i] = new q_action_pair_type *[A[i].size()];
	}

	// q_action_pair_type *min_heaps[S][A_max];
	q_action_pair_type ***min_heaps = new q_action_pair_type **[S];
	for (int i = 0; i < S; ++i)
	{
		// min_heaps[i] = new q_action_pair_type*[A_max];
		min_heaps[i] = new q_action_pair_type *[A[i].size()];
	}

	// The underlying arrays, for each state, of the pairs
	// holds the actual objects, but they never move

	// q_action_pair_type max_heaps_object_arrays[S][A_max];
	q_action_pair_type **max_heaps_object_arrays = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		// max_heaps_object_arrays[i] = new q_action_pair_type[A_max];
		max_heaps_object_arrays[i] = new q_action_pair_type[A[i].size()];
	}

	// q_action_pair_type min_heaps_object_arrays[S][A_max];
	q_action_pair_type **min_heaps_object_arrays = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		// min_heaps_object_arrays[i] = new q_action_pair_type[A_max];
		min_heaps_object_arrays[i] = new q_action_pair_type[A[i].size()];
	}

	// Heap sizes of each state. Last index of heaps of state s is such heap_size[s] - 1
	// only one variable per state, as the heaps has
	int *heap_size = new int[S];

	//**********************************************************************
	// FILL THE HEAPS AS A PREPROCESSING STEP BEFORE ITERATIONS BEGIN
	for (int s = 0; s < S; s++)
	{

		// pointers to the heaps of current state s
		// remember, an array varible is just a pointer to the beginning of the array
		// now, it is an pointer to an array
		q_action_pair_type **max_heap_s = max_heaps[s];
		q_action_pair_type **min_heap_s = min_heaps[s];

		// The underlying array with the actual objects
		// place the objects here
		q_action_pair_type *max_heap_s_object_array = max_heaps_object_arrays[s];
		q_action_pair_type *min_heap_s_object_array = min_heaps_object_arrays[s];

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
			double q_1_U_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);
			q_action_pair_type q_a_pair = make_pair(q_1_U_s_a, a);

			// push pair into max heap and record its initial index in the heap array(vector)
			// seems most natural to put the pair of action a at index a
			max_heap_s_object_array[a] = q_a_pair;
			max_heap_s[initial_index] = &max_heap_s_object_array[a];
			max_ind[a] = initial_index;

			// push pair into min heap and record its initial index in the heap array(vector)
			min_heap_s_object_array[a] = q_a_pair;
			min_heap_s[initial_index] = &min_heap_s_object_array[a];
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
	double *V_U_current_iteration = V_U[0];
	double *V_L_current_iteration = V_L[0];
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
			q_action_pair_type **max_heap_s = max_heaps[s];
			q_action_pair_type **min_heap_s = min_heaps[s];

			// pointers to the indicies arrays of state s
			int *max_heap_indicies_s = heap_max_indicies[s];
			int *min_heap_indicies_s = heap_min_indicies[s];
			//double Q_max = numeric_limits<double>::min();
			double Q_max =-10000;
			// keep the rewards from the upper bound for
			int A_s_size = heap_size[s];
			double Q_U_s[A_s_size];

			// ranged for loop over all actions in the action set of state s
			// This is changed to use the max_heap as an array of actions in A[s]
			for (int a_index = 0; a_index < A_s_size; a_index++)
			{

				// get the action in the a_index of max_heap_s that is used as A[s]
				int a = (*max_heap_s[a_index]).second;

				// reference to the probability vector of size S
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];

				double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_current_iteration, P_s_a_nonzero);

				if (Q_L_s_a > V_L_current_iteration[s])
				{
					V_L_current_iteration[s] = Q_L_s_a;
				}
			}

			// Beginning of updating top upper bound actions
			int heap_max_top_action = (*max_heap_s[0]).second;

			// START UPDATE update_top_action_max, write in seperate function in the future
			q_action_pair_type top_pair = (*max_heap_s[0]);
			int top_pair_action = top_pair.second;

			auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
			double q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_current_iteration, P_s_top_pair_nonzero);

			decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
			int index_in_min_heap = min_heap_indicies_s[top_pair_action];
			decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

			// END UPDATE
			int updated_heap_max_top_action = (*max_heap_s[0]).second;

			while (top_pair_action != updated_heap_max_top_action)
			{
				// Beginning of updating top upper bound actions
				heap_max_top_action = (*max_heap_s[0]).second;

				// START UPDATE update_top_action_max, write in seperate function in the future
				q_action_pair_type top_pair = *max_heap_s[0];
				top_pair_action = top_pair.second;
				auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
				q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_current_iteration, P_s_top_pair_nonzero);

				decrease_max(q_i_s_a, 0, max_heap_s, max_heap_indicies_s, heap_size[s]);
				index_in_min_heap = min_heap_indicies_s[top_pair_action];
				decrease_min(q_i_s_a, index_in_min_heap, min_heap_s, min_heap_indicies_s);

				// END UPDATE
				updated_heap_max_top_action = (*max_heap_s[0]).second;
			}

			// based on the proven fact that the top action now has the maximum value, that is now set
			V_U_current_iteration[s] = (*max_heap_s[0]).first;

			// WE NOW START THE ACTION ELIMINATION PROCESS BASED ON THE TWO HEAPS

			// START UPDATE TOP ACTION MIN TODO: make own function

			q_action_pair_type top_pair_min = *min_heap_s[0];
			int top_pair_action_min = top_pair_min.second; // was a mistake here that it was top_pair and not top_pair_min

			auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
			double q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_current_iteration, P_s_action_min_nonzero);
			(*min_heap_s[0]).first = q_i_s_a_min;
			int index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
			decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

			// END UPDATE TOP ACTION MIN

			// the first candidate value to be considered
			double updated_min_upper_bound = (*min_heap_s[0]).first;

			while (updated_min_upper_bound < V_L_current_iteration[s])
			{

				// REMOVE THE TOP MIN ACTION
				int removableAction = (*min_heap_s[0]).second;

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

				q_action_pair_type top_pair_min = (*min_heap_s[0]);
				top_pair_action_min = top_pair_min.second; // also changed with same mistake
				auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
				q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_current_iteration, P_s_action_min_nonzero);
				(*min_heap_s[0]).first = q_i_s_a_min;
				index_in_max_heap = max_heap_indicies_s[top_pair_action_min];
				decrease_max(q_i_s_a_min, index_in_max_heap, max_heap_s, max_heap_indicies_s, heap_size[s]);

				// END UPDATE TOP ACTION MIN
				updated_min_upper_bound = (*min_heap_s[0]).first;
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
		copy(V_U[(0)], V_U[(0)] + S, result.begin());
	}
	else if (lower_convergence_criteria)
	{
		copy(V_L[(0)], V_L[(0)] + S, result.begin());
	}
	V_type result_tuple = make_tuple(result, iterations, work_per_iteration, actions_eliminated);

	// deallocate the memory
	delete[] heap_size;

	for (int i = 0; i < 1; ++i)
	{
		delete[] V_U[i];
	}
	delete[] V_U;

	for (int i = 0; i < 1; ++i)
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
		delete[] max_heaps_object_arrays[i];
	}
	delete[] max_heaps_object_arrays;

	for (int i = 0; i < S; ++i)
	{
		delete[] min_heaps_object_arrays[i];
	}
	delete[] min_heaps_object_arrays;

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

	return result_tuple;
}