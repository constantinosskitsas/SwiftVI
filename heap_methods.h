#ifndef HEAP_METHODS_H
#define HEAP_METHODS_H

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

int Parent_old(int i);

int Left_old(int i);

int Right_old(int i);

void heapify_max(int i, q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size);

void heapify_min(int i, q_action_pair_type min_heap[], int min_heap_indicies[], int heap_size);

void decrease_max(double newValue, int indexToChange, q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size);

void decrease_min(double newValue, int indexToChange, q_action_pair_type min_heap[], int min_heap_indicies[]);

void remove_index_max_heap(int indexToRemove, q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size);

void remove_index_min_heap(int indexToRemove, q_action_pair_type min_heap[], int min_heap_indicies[], int heap_size);

void build_max_heap(q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size);

void build_min_heap(q_action_pair_type min_heap[], int min_heap_indicies[], int heap_size);

#endif
