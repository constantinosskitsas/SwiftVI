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

int Parent_old(int i)
{
	return (i - 1) / 2;
}

int Left_old(int i)
{
	return 2 * i + 1;
}

int Right_old(int i)
{
	return 2 * i + 2;
}

void heapify_max(int i, q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size)
{

	int largest = i;

	// the left child index and value
	int l = Left_old(i);

	// The right child index and value
	int r = Right_old(i);

	if (l < heap_size && max_heap[l].first > max_heap[i].first)
	{
		largest = l;
	}

	if (r < heap_size && max_heap[r].first > max_heap[largest].first)
	{
		largest = r;
	}

	if (largest != i)
	{
		q_action_pair_type pair_i = max_heap[i];
		q_action_pair_type pair_largest = max_heap[largest];
		max_heap[i] = pair_largest;
		max_heap[largest] = pair_i;
		max_heap_indicies[pair_i.second] = largest;
		max_heap_indicies[pair_largest.second] = i;
		heapify_max(largest, max_heap, max_heap_indicies, heap_size);
	}
}

void heapify_min(int i, q_action_pair_type min_heap[], int min_heap_indicies[], int heap_size)
{

	int smallest = i;

	// the left child index and value
	int l = Left_old(i);

	// The right child index and value
	int r = Right_old(i);

	if (l < heap_size && min_heap[l].first < min_heap[i].first)
	{
		smallest = l;
	}

	if (r < heap_size && min_heap[r].first < min_heap[smallest].first)
	{
		smallest = r;
	}

	if (smallest != i)
	{
		q_action_pair_type pair_i = min_heap[i];
		q_action_pair_type pair_smallest = min_heap[smallest];
		min_heap[i] = pair_smallest;
		min_heap[smallest] = pair_i;
		min_heap_indicies[pair_i.second] = smallest;
		min_heap_indicies[pair_smallest.second] = i;
		heapify_min(smallest, min_heap, min_heap_indicies, heap_size);
	}
}

void decrease_max(double newValue, int indexToChange, q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size)
{
	max_heap[indexToChange].first = newValue;
	heapify_max(indexToChange, max_heap, max_heap_indicies, heap_size);
}

// does not need heap size as the index decreases
void decrease_min(double newValue, int indexToChange, q_action_pair_type min_heap[], int min_heap_indicies[])
{
	min_heap[indexToChange].first = newValue;
	int i = indexToChange;
	while (i > 0 && min_heap[Parent_old(i)].first > min_heap[i].first)
	{
		q_action_pair_type pair_i = min_heap[i];
		q_action_pair_type pair_parent = min_heap[Parent_old(i)];
		min_heap[i] = pair_parent;
		min_heap[Parent_old(i)] = pair_i;
		min_heap_indicies[pair_i.second] = Parent_old(i);
		min_heap_indicies[pair_parent.second] = i;
		i = Parent_old(i);
	}
}

// what if the heap is empty, will not happen in our case, as we cannot with < prune away last action. Safeguard?
void remove_index_max_heap(int indexToRemove, q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size)
{

	// The last index of the array/vector that holds the heap
	int lastElementIndex = heap_size - 1;

	// record the action changed for recording of the heap
	int action_last_element = max_heap[lastElementIndex].second;

	// This can be removed, if not relveant for viewing
	int action_to_remove = max_heap[indexToRemove].second;

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
void remove_index_min_heap(int indexToRemove, q_action_pair_type min_heap[], int min_heap_indicies[], int heap_size)
{

	// The last index of the array/vector that holds the heap
	int lastElementIndex = heap_size - 1;

	// record the action changed for recording of the heap
	int action_last_element = min_heap[lastElementIndex].second;

	// record that the action is now not in A[s] by setting to -1
	// This can be removed, if not relveant for viewing
	int action_to_remove = min_heap[indexToRemove].second;
	min_heap_indicies[action_to_remove] = -1;

	// Set the index to remove to the lastElementIndex
	min_heap[indexToRemove] = min_heap[lastElementIndex];

	// record the change of index in the min_heap_index
	min_heap_indicies[action_last_element] = indexToRemove;

	// printf("10 HERE\n"); //TESTING
	// now heapify the heap from index indexToRemove, which now breaks the heap-property
	heapify_min(indexToRemove, min_heap, min_heap_indicies, heap_size - 1);
}

void build_max_heap(q_action_pair_type max_heap[], int max_heap_indicies[], int heap_size)
{

	// This has implicit floor in the division, which is the correct way
	int index_of_last_non_leaf_node = (heap_size / 2) - 1;

	for (int i = index_of_last_non_leaf_node; i >= 0; i--)
	{
		heapify_max(i, max_heap, max_heap_indicies, heap_size);
	}
}

void build_min_heap(q_action_pair_type min_heap[], int min_heap_indicies[], int heap_size)
{

	// This has implicit floor in the division, which is the correct way
	int index_of_last_non_leaf_node = (heap_size / 2) - 1;

	for (int i = index_of_last_non_leaf_node; i >= 0; i--)
	{
		heapify_min(i, min_heap, min_heap_indicies, heap_size);
	}
}
