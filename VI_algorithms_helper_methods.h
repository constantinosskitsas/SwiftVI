#ifndef VI_ALGORITHMS_HELPER_METHODS_H
#define VI_ALGORITHMS_HELPER_METHODS_H

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

double abs_max_diff(double V_one[], double V_two[], int S);

double sum_of_mult_nonzero_only(const vector<double> &V_one, double V_two[], const vector<int> &non_zero_transition_states);

double sum_of_mult(const vector<double> &V_one, double V_two[]);

double find_max_R(const R_type &R);

vector<double> find_max_R_for_each_state(const R_type &R);

tuple<double, double, vector<double>> find_all_r_values(const R_type &R);

double find_min_R(const R_type &R);

V_result_type V_upper_lower_average(double V_one[], double V_two[], int S);

int find_max_A(const A_type &A);

void does_heap_and_indicies_match(heap_of_pairs_type heap, int indicies[], int A_max);

void are_heaps_synced(heap_of_pairs_type &max_heap, heap_of_pairs_type &min_heap);

A_type copy_A(const A_type &A);

double abs_max_diff_vectors(const V_result_type &V_one, const V_result_type &V_two);
double sum_of_mult_nonzero_only1(const vector<double> &V_one, double V_two[], const vector<int> &non_zero_transition_states);
pair<double,double> sum_of_mult_nonzero_onlyT(const vector<double> &V_one, double V_two[],double V_three[], const vector<int> &non_zero_transition_states);

#endif
