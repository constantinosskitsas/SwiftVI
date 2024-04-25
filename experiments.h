#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

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

void runMBIE(MDP_type mdp, int S, int nA);
void RLRS(string filename, int expnum, int States, int Actions, int SS, int StartP, int endP, int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition);
void GSTM(string filename, int expnum, int States, int Actions, int SS, int StartP, int endP, int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition);
void REXP(string filename, int expnum, int States, int Actions, int SS, int StartP, int endP, int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition);
void VMS(int NOFexp, double epsilon, double gamma);
void write_stringstream_to_file(ostringstream &string_stream, ofstream &output_stream, string file_name);
void create_data_tables_number_of_statesGSAll(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition);

void create_data_tables_action_prob(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition);

void create_data_tables_number_of_actions(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition);

void create_data_tables_rewards(string filename, int S, int A_num, double epsilon, double gamma, double reward_factor, double reward_prob, double upper_reward, double action_prob, double non_zero_transition);

void create_data_tables_transition_prob(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward);

void create_data_tables_number_of_states(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition);

void create_data_tables_number_of_states_and_actions(string filename, int A_S_max, double epsilon, double gamma, double upper_reward, double non_zero_transition);

void create_data_tables_number_of_states_iterations(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition);

void create_data_tables_number_of_transitions(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, int max_transitions);

void create_data_tables_number_of_actions_iterations(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition);

void create_data_tables_max_reward(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, double max_reward_finishing_value, double non_zero_transition);

void create_data_tables_number_of_actions_first_convergence_iteration(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition);

void create_data_tables_work_per_iteration(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int non_zero_transition, double mean, double variance);

void create_data_tables_top_action_change(string filename, int S, int A_num, double epsilon, double gamma, double upper_reward, double action_prob, int number_of_transitions, double mean, double variance, double lambda);

void create_data_tables_normal_dist_varying_variance(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int transition_list_size, double mean, double min_variance, double max_variance, double variance_increment);

void create_data_tables_VIH_distributions_iterations_work(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean_1, double mean_2, double mean_3, double variance_1, double variance_2, double variance_3, double lambda_1, double lambda_2, double lambda_3, double upper_reward_1, double upper_reward_2, double upper_reward_3);

void create_data_tables_exponential_dist_varying_lambda(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int transition_list_size, double min_lambda, double max_lambda, double lambda_increment);

void create_data_tables_work_per_iteration_VIAEH_implementations(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean, double variance);

void create_data_tables_bounds_comparisons(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean, double variance);

void create_data_tables_actions_touched(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int number_of_transitions, double mean, double variance);

void create_data_tables_work_per_iteration_BEST_implementations(string filename, int S, int A_num, double epsilon, double gamma, double action_prob, int non_zero_transition, double mean, double variance);
void create_data_tables_number_of_actions_best_implementations(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition);
void create_data_tables_number_of_states_best_implementations(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition);

void create_data_tables_number_of_actions_VIH_implementations(string filename, int S, int A_max, double epsilon, double gamma, double upper_reward, double non_zero_transition);
void create_data_tables_VMS(string filename,double epsilon,double gamma);
void create_data_tables_number_of_statesGS(string filename, int S_max, int A_num, double epsilon, double gamma, double upper_reward, double non_zero_transition);
void create_data_tables_number_GS(string filename,int expnum,int States,int Actions,int SS,int startP,int endP,int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition);
void create_data_tables_number_GSBO(string filename,int expnum,int States,int Actions,int SS,int startP,int endP,int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition);
void create_data_tables_number_GSTM(string filename,int expnum,int States,int Actions,int SS,int StartP,int endP,int IncP, double epsilon, double gamma, double upper_reward, double non_zero_transition);

#endif
