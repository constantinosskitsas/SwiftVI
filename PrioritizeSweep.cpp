#include <queue>
#include <functional>

#include <vector>
#include <utility> // for std::pair
#include <tuple>
 // for std::function
#include <chrono> // for microseconds
#include "PrioritizeSweep.h"

/*bool cmp = [](std::pair<double, int> a, std::pair<double, int> b) {
		if (a.first == b.first) {
			return a.second > b.second; 
		} else {
			return a.first < b.first;
	}
    };*/
void performIteration(int S, A_type &A, R_type &R, P_type &P, double gamma, double* V_current_iteration,
                      std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, ComparatorType>& PriorityHeap,
                      std::vector<int>& policy, std::vector<std::vector<int>>& predecessor,
                      double* reverseV) {
    for (int s = 0; s < S; s++) {
        double oldV = V_current_iteration[s];

        for (auto a : A[s]) {
            auto &[P_s_a, P_s_a_nonzero] = P[s][a];
            double cum_sum = 0.0;
            int k = 0;

            for (int ks : P_s_a_nonzero) {
                cum_sum += (P_s_a[k] * V_current_iteration[ks]);
                k++;
                if (P_s_a[k] > 0.0009) {
                    if (predecessor[ks].empty())
                        predecessor[ks].push_back(s);
                    else if (predecessor[ks].back() != s)
                        predecessor[ks].push_back(s);
                }
            }

            double R_s_a = R[s][a] + gamma * cum_sum;
            if (R_s_a > V_current_iteration[s]) {
                V_current_iteration[s] = R_s_a;
                policy[s] = a;
            }
        }
        PriorityHeap.push({V_current_iteration[s]-oldV , s});
        reverseV[s] = V_current_iteration[s]-oldV;
    }
}


void performIterationPred(int s, A_type &A, R_type &R, P_type &P, double gamma, double* V_current_iteration,
                      std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, ComparatorType>& PriorityHeap,
                      std::vector<int>& policy, std::vector<std::vector<int>>& predecessor,
                      double* reverseV,double convergence_bound_precomputed)
        {
           
            for (auto sa: predecessor[s]){
				double oldV = V_current_iteration[sa];
			// ranged for loop over all actions in the action set of state s
			    for (auto a : A[sa])
			    {
				    auto &[P_s_a, P_s_a_nonzero] = P[sa][a];
				    double R_s_a = R[sa][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_current_iteration, P_s_a_nonzero);
				    if (R_s_a > V_current_iteration[sa])
				    {
					    V_current_iteration[sa] = R_s_a;
					    policy[sa] = a;
				    }
			    }
			    if(abs(oldV-V_current_iteration[sa])>convergence_bound_precomputed){
				    PriorityHeap.push({V_current_iteration[sa]-oldV,sa});
				    reverseV[sa]=V_current_iteration[sa]-oldV;
		        }
			}
        }

void performIterationUP(int S, A_type &A, R_type &R, P_type &P, double gamma, double* V_current_iteration,
                      std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, ComparatorType>& PriorityHeap,
                      std::vector<int>& policy, std::vector<std::vector<int>>& predecessor,
                      double* reverseV) {
    double Q_max;
    for (int s = 0; s < S; s++) {
        double oldV = V_current_iteration[s];
    Q_max = -100000;
        for (auto a : A[s]) {
            auto &[P_s_a, P_s_a_nonzero] = P[s][a];
            double cum_sum = 0.0;
            int k = 0;

            for (int ks : P_s_a_nonzero) {
                cum_sum += (P_s_a[k] * V_current_iteration[ks]);
                k++;
                if (P_s_a[k] > 0.0009) {
                    if (predecessor[ks].empty())
                        predecessor[ks].push_back(s);
                    else if (predecessor[ks].back() != s)
                        predecessor[ks].push_back(s);
                }
            }

            double R_s_a = R[s][a] + gamma * cum_sum;
            if (R_s_a > Q_max)
				{
					Q_max = R_s_a;
                    policy[s] = a;
				}
        }
         V_current_iteration[s]=Q_max;
        PriorityHeap.push({oldV - V_current_iteration[s], s});
        reverseV[s] = oldV - V_current_iteration[s];
    }
}

void performIterationPredUP(int s, A_type &A, R_type &R, P_type &P, double gamma, double* V_current_iteration,
                      std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, ComparatorType>& PriorityHeap,
                      std::vector<int>& policy, std::vector<std::vector<int>>& predecessor,
                      double* reverseV,double convergence_bound_precomputed)
        {
           double Q_max;
            for (auto sa: predecessor[s]){
                Q_max = -100000;
				double oldV = V_current_iteration[sa];
			// ranged for loop over all actions in the action set of state s
			    for (auto a : A[sa])
			    {
				    auto &[P_s_a, P_s_a_nonzero] = P[sa][a];
				    double R_s_a = R[sa][a] + gamma * sum_of_mult_nonzero_only(P_s_a, V_current_iteration, P_s_a_nonzero);
            if (R_s_a > Q_max)
				{
					Q_max = R_s_a;
                    policy[sa] = a;
				}
			    }
                V_current_iteration[sa]=Q_max;
			    if(abs(oldV-V_current_iteration[sa])>convergence_bound_precomputed){
				    PriorityHeap.push({oldV-V_current_iteration[sa],sa});
				    reverseV[sa]=oldV-V_current_iteration[sa];
		        }
			}
        }