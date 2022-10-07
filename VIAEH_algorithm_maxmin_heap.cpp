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
#include <math.h>

using namespace std;
using namespace std::chrono;

int Parent_maxmin(int i)
{
	return (i - 1) / 2;
}

int Left_maxmin(int i)
{
	return 2 * i + 1;
}

int Right_maxmin(int i)
{
	return 2 * i + 2;
}

bool is_internal_node(int index, int heap_size)
{
	return index <= (heap_size / 2) - 1;
}

// assume heap is not empty, which is the case with the logic in the problem
int get_max_action(q_action_pair_type max_min_heap[])
{
	return max_min_heap[0].second;
}

// assume heap is not empty, which is the case with the logic in the problem
double get_max_q_value(q_action_pair_type max_min_heap[])
{
	return max_min_heap[0].first;
}

// special case, when the heap has size 1 and 2, i.e. there can be 0, 1, or 2 elements on level 1, the first min level
// return 0 if heap has size 1, and max element is also min element
// return 1 if min element is left child of root
// return 2 if min element is right child of root
// these cases are actually not edge cases, as we hope to eliminate so many actions such that the heap consists of only 1 or 2 actions
// return q_value, action, index
tuple<double, int, int> get_min_q_value_action_index(q_action_pair_type max_min_heap[], int heap_size)
{
	if (heap_size == 1)
	{
		return make_tuple(max_min_heap[0].first, max_min_heap[0].second, 0);
	}
	else if (heap_size == 2)
	{
		// has then to be this element per max-min-heap property
		return make_tuple(max_min_heap[1].first, max_min_heap[1].second, 1);
	}
	else
	{
		// we have 2 elements on min row
		if (max_min_heap[1].first < max_min_heap[2].first)
		{
			return make_tuple(max_min_heap[1].first, max_min_heap[1].second, 1);
		}
		else
		{
			return make_tuple(max_min_heap[2].first, max_min_heap[2].second, 2);
		}
	}
}

void push_down_min_iter(int index, q_action_pair_type max_min_heap[], int heap_size)
{
	// start with first call argument index
	int m = index;

	// while m is not index of a leaf, i.e. it is an internal node with at least one leaf
	while (m <= (heap_size / 2) - 1)
	{
		// find smallest child or grandchild of i
		// start with Left_maxmin(i), as it is surely there as i is not an leaf
		int i = m;

		// in loop, m is the index of smallest child or grandchild found so far
		m = Left_maxmin(i);
		for (int descendant_index : {2 * i + 2, 4 * i + 3, 4 * i + 4, 4 * i + 5, 4 * i + 6})
		{
			// if index is not in heap anymore, then we have considered all children and granchildren
			// we therefore break out of the loop
			if (descendant_index >= heap_size)
			{
				break;
			}
			// if we have a new smallest index
			if (max_min_heap[descendant_index].first < max_min_heap[m].first)
			{
				m = descendant_index;
			}
		}
		if (max_min_heap[m].first < max_min_heap[i].first)
		{
			swap(max_min_heap[m], max_min_heap[i]);

			// if m is a granchild of i
			// check if it is NOT a child, then it is a grandchild! two less comparisons
			if ((m != 2 * i + 1) && (m != 2 * i + 2) && (max_min_heap[m].first > max_min_heap[Parent_maxmin(m)].first))
			{
				swap(max_min_heap[m], max_min_heap[Parent_maxmin(m)]);
			}
		}
		else
		{
			// it is in the right place, so do not "recurse" anymore
			break;
		}
	}
}

void push_down_min(int i, q_action_pair_type max_min_heap[], int heap_size)
{
	// if i is not index of a leaf, i.e. it is an internal node with at least one leaf
	if (i <= (heap_size / 2) - 1)
	{
		// find smallest child or grandchild of i
		// start with Left_maxmin(i), as it is surely there as i is not an leaf
		int m = Left_maxmin(i);
		for (int descendant_index : {2 * i + 2, 4 * i + 3, 4 * i + 4, 4 * i + 5, 4 * i + 6})
		{
			// if index is not in heap anymore, then we have considered all children and granchildren
			// we therefore break out of the loop
			if (descendant_index >= heap_size)
			{
				break;
			}
			// if we have a new smallest index
			if (max_min_heap[descendant_index].first < max_min_heap[m].first)
			{
				m = descendant_index;
			}
		}
		// m is not a child of i, i.e. it is a grandchild
		if ((m != 2 * i + 1) && (m != 2 * i + 2))
		{
			if (max_min_heap[m].first < max_min_heap[i].first)
			{
				swap(max_min_heap[m], max_min_heap[i]);
				if (max_min_heap[m].first > max_min_heap[Parent_maxmin(m)].first)
				{
					swap(max_min_heap[m], max_min_heap[Parent_maxmin(m)]);
				}
				push_down_min(m, max_min_heap, heap_size);
			}
		}
		else
		{
			if (max_min_heap[m].first < max_min_heap[i].first)
			{
				swap(max_min_heap[m], max_min_heap[i]);
			}
		}
	}
}

void push_down_max(int i, q_action_pair_type max_min_heap[], int heap_size)
{
	// if i is not index of a leaf, i.e. it is an internal node with at least one leaf
	if (i <= (heap_size / 2) - 1)
	{
		// find smallest child or grandchild of i
		// start with Left_maxmin(i), as it is surely there as i is not an leaf
		int m = Left_maxmin(i);
		for (int descendant_index : {2 * i + 2, 4 * i + 3, 4 * i + 4, 4 * i + 5, 4 * i + 6})
		{
			// if index is not in heap anymore, then we have considered all children and granchildren
			// we therefore break out of the loop
			if (descendant_index >= heap_size)
			{
				break;
			}
			// if we have a new smallest index
			if (max_min_heap[descendant_index].first > max_min_heap[m].first)
			{
				m = descendant_index;
			}
		}
		// m is not a child of i, i.e. it is a grandchild
		if ((m != 2 * i + 1) && (m != 2 * i + 2))
		{
			if (max_min_heap[m].first > max_min_heap[i].first)
			{
				swap(max_min_heap[m], max_min_heap[i]);
				if (max_min_heap[m].first < max_min_heap[Parent_maxmin(m)].first)
				{
					swap(max_min_heap[m], max_min_heap[Parent_maxmin(m)]);
				}
				push_down_max(m, max_min_heap, heap_size);
			}
		}
		else
		{
			if (max_min_heap[m].first > max_min_heap[i].first)
			{
				swap(max_min_heap[m], max_min_heap[i]);
			}
		}
	}
}

void push_down_max_iter(int index, q_action_pair_type max_min_heap[], int heap_size)
{
	// start with first call argument index
	int m = index;

	// while m is not index of a leaf, i.e. it is an internal node with at least one leaf
	while (m <= (heap_size / 2) - 1)
	{
		// find largest child or grandchild of i
		// start with Left_maxmin(i), as it is surely there as i is not an leaf
		int i = m;

		// in loop, m is the index of smallest child or grandchild found so far
		m = Left_maxmin(i);
		for (int descendant_index : {2 * i + 2, 4 * i + 3, 4 * i + 4, 4 * i + 5, 4 * i + 6})
		{
			// if index is not in heap anymore, then we have considered all children and granchildren
			// we therefore break out of the loop
			if (descendant_index >= heap_size)
			{
				break;
			}
			// if we have a new largest index
			if (max_min_heap[descendant_index].first > max_min_heap[m].first)
			{
				m = descendant_index;
			}
		}
		if (max_min_heap[m].first > max_min_heap[i].first)
		{
			swap(max_min_heap[m], max_min_heap[i]);

			// if m is a granchild of i
			// check if it is NOT a child, then it is a grandchild! two less comparisons
			if ((m != 2 * i + 1) && (m != 2 * i + 2) && (max_min_heap[m].first < max_min_heap[Parent_maxmin(m)].first))
			{
				swap(max_min_heap[m], max_min_heap[Parent_maxmin(m)]);
			}
		}
		else
		{
			// it is in the right place, so do not "recurse" anymore
			break;
		}
	}
}

// decreasing the min value does not break the heap property, and nothing should be reordered
void decrease_min(double newValue, int min_index, q_action_pair_type max_min_heap[])
{
	max_min_heap[min_index].first = newValue;
}

// decrease the max value, and possible bubble it down, as another value can now be the max value
void decrease_max_element_max_min_heap(double newValue, q_action_pair_type max_min_heap[], int heap_size)
{
	max_min_heap[0].first = newValue;
	push_down_max(0, max_min_heap, heap_size);
}

// what if the heap is empty, will not happen in our case, as we cannot with < prune away last action. Safeguard?
// DOES NOT change heap_size, have to do that outside function
void remove_min_element_max_min_heap(int indexToRemove, q_action_pair_type max_min_heap[], int heap_size)
{
	// The last index of the array/vector that holds the heap
	// int lastElementIndex = heap_size - 1;

	swap(max_min_heap[indexToRemove], max_min_heap[heap_size - 1]);

	// you always delete one of the two min elements on level 1, so always push_down_min that is to be called
	// what if you delete the last index of the heap, i.e. the size of the heap is 2 or 3?
	// in these special cases, do nothing! The heap property is not broken
	// TODO: double check that heap size 2 and 3 has heap property if min is deleted here!
	if (heap_size > 3)
	{
		push_down_min(indexToRemove, max_min_heap, heap_size - 1);
	}
}

// Push-down defines that it is a max-min heap, in that a odd level is a min level. Thus level 0, which is consideren even, is a max level.
void push_down(int i, q_action_pair_type max_min_heap[], int heap_size)
{
	// If odd level, then it is a min level in a max-min heap
	// find more efficient way to find level of node in complete binary tree, perhaps from CP
	// should be correct now
	int level = int(floor(log2(double(i + 1))));
	if (level % 2 == 1)
	{
		push_down_min(i, max_min_heap, heap_size);
	}
	else
	{
		push_down_max(i, max_min_heap, heap_size);
	}
}

void build_max_min_heap(q_action_pair_type max_min_heap[], int heap_size)
{
	// This has implicit floor in the division, which is the correct way
	// int index_of_last_non_leaf_node = (heap_size / 2) - 1;
	for (int i = (heap_size / 2) - 1; i >= 0; i--)
	{
		// TODO changed from this, is it heap_size or i that is the size?
		// push_down(i, max_min_heap, heap_size);
		push_down(i, max_min_heap, heap_size);
	}
}

V_type value_iteration_action_elimination_heaps_max_min_heap(S_type S, R_type R, A_type A, P_type P, double gamma, double epsilon)
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

	// HEAP INITIALIZATION
	q_action_pair_type **max_min_heaps = new q_action_pair_type *[S];
	for (int i = 0; i < S; ++i)
	{
		max_min_heaps[i] = new q_action_pair_type[A_max];
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
		q_action_pair_type *max_min_heap_s = max_min_heaps[s];

		// The initial index in both heaps and the size of the heap
		int initial_index = 0;

		// initialise the heaps with values from the first iteration
		for (int a : A[s])
		{

			// need the distribution to calculate first q-values
			auto &[P_s_a, P_s_a_nonzero] = P[s][a];

			// use the even iteration, as this is the one used in the i = 1 iteration, that we want to pre-do
			double q_1_U_s_a = V_U[0][s]; // R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_U[0], P_s_a_nonzero);
			q_action_pair_type q_a_pair = make_pair(q_1_U_s_a, a);

			// push pair into max heap and record its initial index in the heap array(vector)
			max_min_heap_s[initial_index] = q_a_pair;

			// increment as last step as to use it as index at first and ends up being the size of the heap
			initial_index = initial_index + 1;
		}

		// set the heap size to correct size
		heap_size[s] = initial_index;

		// MAKE THE ARRAYS BE MAX HEAPS WITH MY OWN HEAP ALGORITHM
		// Make the arrays have the heap property
		build_max_min_heap(max_min_heap_s, initial_index);
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
			// V_U_current_iteration[s] = numeric_limits<double>::min();
			V_L_current_iteration[s] = numeric_limits<double>::min();

			// The max heap for the state s
			q_action_pair_type *max_min_heap_s = max_min_heaps[s];

			// ranged for loop over all actions in the action set of state s
			// This is changed to use the max_min_heap as an array of actions in A[s]
			for (int a_index = 0; a_index < heap_size[s]; a_index++)
			{

				// get the action in the a_index of max_heap_s that is used as A[s]
				int a = max_min_heap_s[a_index].second;

				// reference to the probability vector of size S
				auto &[P_s_a, P_s_a_nonzero] = P[s][a];

				double Q_L_s_a = R[s][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_L_previous_iteration, P_s_a_nonzero);

				if (Q_L_s_a > V_L_current_iteration[s])
				{
					V_L_current_iteration[s] = Q_L_s_a;
				}
			}

			// START UPDATE update_top_action_max, write in seperate function in the future
			int top_pair_action = get_max_action(max_min_heap_s);

			// update new value of top action
			auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
			double q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_previous_iteration, P_s_top_pair_nonzero);

			// update the top value with the new updated value
			// posible bubbles it down to make a new action and value the top action
			decrease_max_element_max_min_heap(q_i_s_a, max_min_heap_s, heap_size[s]);

			// END UPDATE
			int updated_heap_max_top_action = get_max_action(max_min_heap_s);

			// TODO: mistake likely in here
			while (top_pair_action != updated_heap_max_top_action)
			{
				// START UPDATE update_top_action_max, write in seperate function in the future
				top_pair_action = get_max_action(max_min_heap_s);

				// update new value of top action
				auto &[P_s_top_pair, P_s_top_pair_nonzero] = P[s][top_pair_action];
				q_i_s_a = R[s][top_pair_action] + gamma * sum_of_mult_nonzero_only(P_s_top_pair, V_U_previous_iteration, P_s_top_pair_nonzero);

				// update the top value with the new updated value
				// posible bubbles it down to make a new action and value the top action
				decrease_max_element_max_min_heap(q_i_s_a, max_min_heap_s, heap_size[s]);

				// END UPDATE
				updated_heap_max_top_action = get_max_action(max_min_heap_s);
			}

			// based on the proven fact that the top action now has the maximum value, that is now set
			V_U_current_iteration[s] = get_max_q_value(max_min_heap_s);

			// WE NOW START THE ACTION ELIMINATION PROCESS BASED ON THE TWO HEAPS

			// START UPDATE TOP ACTION MIN TODO: make own function
			auto [value, top_pair_action_min, min_index] = get_min_q_value_action_index(max_min_heap_s, heap_size[s]);

			// calculate the updated min_action value
			auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
			double q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero);

			// update the min value, where the index returned above is used
			decrease_min(q_i_s_a_min, min_index, max_min_heap_s);

			// the first candidate value to be considered
			// if the while loop condition is so that the min action is to be removed
			// these values are the ones to be used, as this is the action to be removed
			double updated_min_upper_bound;
			int updated_top_pair_action_min;
			int updated_min_index;
			tie(updated_min_upper_bound, updated_top_pair_action_min, updated_min_index) = get_min_q_value_action_index(max_min_heap_s, heap_size[s]);

			while (updated_min_upper_bound < V_L_current_iteration[s])
			{

				// REMOVE THE TOP MIN ACTION
				// record action eliminated to return
				actions_eliminated_in_iteration.emplace_back(s, updated_top_pair_action_min);

				// remove it in the min heap and decrease the heap value
				remove_min_element_max_min_heap(updated_min_index, max_min_heap_s, heap_size[s]);
				heap_size[s]--;

				// START UPDATE TOP ACTION MIN TODO: make own function
				auto [value, top_pair_action_min, min_index] = get_min_q_value_action_index(max_min_heap_s, heap_size[s]);

				// calculate the updated min_action value
				auto &[P_s_action_min, P_s_action_min_nonzero] = P[s][top_pair_action_min];
				q_i_s_a_min = R[s][top_pair_action_min] + gamma * sum_of_mult_nonzero_only(P_s_action_min, V_U_previous_iteration, P_s_action_min_nonzero);

				// update the min value, where the index returned above is used
				decrease_min(q_i_s_a_min, min_index, max_min_heap_s);

				// the first candidate value to be considered
				// if the while loop condition is so that the min action is to be removed
				// these values are the ones to be used, as this is the action to be removed
				tie(updated_min_upper_bound, updated_top_pair_action_min, updated_min_index) = get_min_q_value_action_index(max_min_heap_s, heap_size[s]);
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
		delete[] max_min_heaps[i];
	}
	delete[] max_min_heaps;

	delete[] heap_size;

	return result_tuple;
}
