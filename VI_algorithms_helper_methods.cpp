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
#include <thread>
//#include <execution>

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


double abs_min_diff(std::vector<double> &V_one, std::vector<double> &V_two, int S)
{
	double abs_min = double(0);
	for (int i = 0; i < S; ++i)
	{
		abs_min = min(abs_min, abs(V_one[i] - V_two[i]));
	}
	return abs_min;
}

// Returns the maximum absolute difference between entries in the two vectors
// pass the arguments by reference for efficiency, use const?
double abs_max_diff(std::vector<double> &V_one, std::vector<double> &V_two, int S)
{
	double abs_max = double(0);
	for (int i = 0; i < S; ++i)
	{
		abs_max = max(abs_max, abs(V_one[i] - V_two[i]));
	}
	return abs_max;
}

// Returns the maximum absolute difference between entries in the two vectors
// pass the arguments by reference for efficiency, use const?
double abs_max_diff(double V_one[], double V_two[], int S)
{
	double abs_max = double(0);
	for (int i = 0; i < S; ++i)
	{
		abs_max = max(abs_max, abs(V_one[i] - V_two[i]));
	}
	return abs_max;
}

// define argument references as const, as they are not changed
double sum_of_mult_nonzero_only1(const vector<double> &V_one, double V_two[], const vector<int> &non_zero_transition_states)
{
	double cum_sum = double(0);
	int k = 0;
	for (int s : non_zero_transition_states)
	{
		cum_sum += (V_one[s] * V_two[s]);
		// cum_sum += 	(V_one[k] * V_two[s]);
		// k++;
	}
	return cum_sum;
}
double sum_of_mult_nonzero_only(const vector<double> &V_one, double V_two[], const vector<int> &non_zero_transition_states)
{
	double cum_sum = double(0);
	int k = 0;
	for (int s : non_zero_transition_states)
	{
		// cum_sum += 	(V_one[s] * V_two[s]);
		cum_sum += (V_one[k] * V_two[s]);
		k++;
	}
	return cum_sum;
}
pair<double, double> sum_of_mult_nonzero_onlyT(const vector<double> &V_one, double V_two[], double V_three[], const vector<int> &non_zero_transition_states)
{
	double cum_sum = double(0);
	double cum_sum1 = double(0);
	int k = 0;
	for (int s : non_zero_transition_states)
	{
		// cum_sum += 	(V_one[s] * V_two[s]);
		cum_sum += (V_one[k] * V_two[s]);
		cum_sum1 += (V_one[k] * V_three[s]);
		k++;
	}
	return make_pair(cum_sum, cum_sum1);
}

// define argument references as const, as they are not changed
double sum_of_mult(const vector<double> &V_one, double V_two[])
{
	double cum_sum = double(0);
	for (int i = 0; i < V_one.size(); ++i)
	{
		cum_sum += (V_one[i] * V_two[i]);
	}
	return cum_sum;
}


/********************MBIE*************/

static void sum_mult_segment(const vector<double> &V_one, const vector<double> &V_two, vector<double> &result, const int start, const int end, const int t) {
	for (int i = start; i < end; i++)
	{
		result[t] += (V_one[i] * V_two[i]);
	}
}

double parallel_sum_of_mult(const vector<double> &V_one, const vector<double>  &V_two) {
	int num_threads = std::thread::hardware_concurrency(); // Get the number of threads supported by the system
	std::vector<std::thread> threads(num_threads);
	int chunk_size = V_one.size() / num_threads; // Determine the size of the segment each thread will process
	std::vector<double> result(num_threads, 0.0);
	for (int t = 0; t < num_threads; t++) {
		int start = t * chunk_size;
		int end = (t == num_threads - 1) ? V_one.size() : start + chunk_size; // Ensure the last thread covers the remaining elements
		threads[t] = std::thread(sum_mult_segment,std::cref(V_one),std::cref(V_two),std::ref(result), start, end, t);
	}
	// Join the threads with the main thread
	for (auto& thread : threads) {
		thread.join();
	}
	return std::accumulate(result.begin(),result.end(), 0.0);
}


// used in MBIE
double sum_of_mult(const vector<double> &V_one, const vector<double>  &V_two)
{
	double cum_sum = 0.0;
	for (int i = 0; i < V_one.size(); i++)
	{
		cum_sum += (V_one[i] * V_two[i]);
	}
	return cum_sum;
}

double find_max_R(const R_type &R)
{
	double max_R = numeric_limits<double>::min();
	for (int i = 0; i < R.size(); i++)
	{
		for (int j = 0; j < R[i].size(); j++)
		{
			max_R = max(max_R, R[i][j]);
		}
	}
	return max_R;
}

vector<double> find_max_R_for_each_state(const R_type &R)
{
	int S = R.size();
	vector<double> max_R_for_each_state(S);

	// For each state
	for (int i = 0; i < S; i++)
	{
		double max_R = numeric_limits<double>::min();
		for (int j = 0; j < R[i].size(); j++)
		{
			max_R = max(max_R, R[i][j]);
		}
		max_R_for_each_state[i] = max_R;
	}
	return max_R_for_each_state;
}

tuple<double, double, vector<double>> find_all_r_values(const R_type &R)
{
	int S = R.size();
	vector<double> max_R_for_each_state(S);
	double r_star_min = numeric_limits<double>::max();
	double r_star_max = numeric_limits<double>::min();
	// r_star_min=100;
	// r_star_max=-100;
	// For each state
	for (int i = 0; i < S; i++)
	{
		// find max reward in each state
		// double max_R = numeric_limits<double>::min();
		double max_R = -100000;
		for (int j = 0; j < R[i].size(); j++)
		{
			max_R = max(max_R, R[i][j]);
		}

		// test if it is the r_* value among states so far
		r_star_min = min(r_star_min, max_R);

		// test if largest among states so far
		r_star_max = max(r_star_max, max_R);

		max_R_for_each_state[i] = max_R;
	}
	return make_tuple(r_star_min, r_star_max, max_R_for_each_state);
}

double find_min_R(const R_type &R)
{
	double min_R = numeric_limits<double>::max();
	for (int i = 0; i < R.size(); i++)
	{
		for (int j = 0; j < R[i].size(); j++)
		{
			min_R = min(min_R, R[i][j]);
		}
	}
	return min_R;
}

vector<double> V_upper_lower_average(double V_one[], double V_two[], int S)
{
	vector<double> answer(S, 0);
	for (int i = 0; i < S; ++i)
	{
		double average_of_index_i = (V_one[i] + V_two[i]) / double(2);
		answer[i] = average_of_index_i;
	}
	return answer;
}

// Find the maximum action to have this as the maximum action to keep indicies for in every state
int find_max_A(const A_type &A)
{
	int max_A = 0;
	for (int i = 0; i < A.size(); i++)
	{
		for (int j = 0; j < A[i].size(); j++)
		{
			max_A = max(max_A, A[i][j]);
		}
	}
	return max_A;
}

void does_heap_and_indicies_match(heap_of_pairs_type heap, int indicies[], int A_max)
{
	for (int i = 0; i < A_max; i++)
	{
		if (indicies[i] != -1)
		{
			if (heap[indicies[i]].second != i)
			{
				printf("THEY DO NOT MATCH!\n");
			}
		}
	}
}

// check if same size first!
void are_heaps_synced(heap_of_pairs_type &max_heap, heap_of_pairs_type &min_heap)
{
	if (max_heap.size() != min_heap.size())
	{
		printf("THEY ARE NOT THE SAME SIZE\n");
		printf("max\n");
		print_heap(max_heap);
		printf("min\n");
		print_heap(min_heap);
	}
	for (auto p_max : max_heap)
	{
		for (auto p_min : min_heap)
		{
			if (p_max.second == p_min.second)
			{
				if (p_max.first != p_min.first)
				{
					printf("THE TWO HEAPS ARE NOT SYNCED!\n");
				}
			}
		}
	}
}

A_type copy_A(const A_type &A)
{
	A_type A_copy;
	for (auto a_s : A)
	{
		A_copy.push_back(a_s);
	}
	return A_copy;
}

double abs_max_diff_vectors(const V_result_type &V_one, const V_result_type &V_two)
{
	double abs_max = double(0);
	for (int i = 0; i < V_one.size(); ++i)
	{
		abs_max = max(abs_max, abs(V_one[i] - V_two[i]));
		// if(abs(V_one[i] - V_two[i])>0.1)
		//	cout<<"prob in"<<i<<V_one[i]<<" , "<<V_two[i]<<endl;
	}
	return abs_max;
}
