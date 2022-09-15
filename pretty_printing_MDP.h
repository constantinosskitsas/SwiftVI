#ifndef PRETTY_PRINTING_MDP_H
#define PRETTY_PRINTING_MDP_H

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

void print_R(const R_type &R);

void print_A(const A_type &A);

void print_P(const P_type &P);

void print_V(const V_result_type &V);

void print_V_array(double arr[], int arr_size);

void print_heap(heap_of_pairs_type heap);

void print_max_min_heap(q_action_pair_type *max_min_heap, int heap_size);

void print_int_array(int arr[], int arr_size);
#endif
