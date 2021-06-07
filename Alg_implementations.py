# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:23:45 2021

@author: priya
"""
import numpy as np
import copy
import time
import random
import networkx as nx
from csv import reader
from collections import OrderedDict
import matplotlib.pyplot as plt

#Input files:
#Network file original - Used in section 1
network_file_path = "Braess_net.txt"
network_flow_weight_file_path = "Braess_partial_interaction.csv"
OD_file_path = "Braess_trips.txt"

# network_file_path = "SF_net.txt"
# OD_file_path = "SF_trips.txt"

# network_file_path = "EMA_net.txt"
# OD_file_path = "EMA_trips.csv"


#Constants defined here
random.seed(1)
n = 9
G = nx.DiGraph()
start_time = time.time()
num_MSA_iterations = 100
num_FW_iterations = 100
num_GP_iterations = 100

##################Section 1############################
#This block reads in the network file. It also separates the link info for networkx graph
with open(network_file_path) as f:
    net = f.readlines()
f.close()
del f
#This calculates the number of lines in the network file.
#It is assumed that there are 4 lines of metadata, one line end of metadata, 2 lines of space and a line of headings
num_lines_datafile = len(net)
num_links = num_lines_datafile-n
del num_lines_datafile

net_data = net[n:]
del net
final_net_data =[]

link_data=[]
capacity_data=[]
FFT_data=[]
alpha_data=[]
beta_data=[]
link_data_copy=[]
RG_array = []

for i in range(num_links):
    a = net_data[i].split('\t')
    final_net_data.append(a)
    q, b, c = a[:3]
    link=(b,c)
    cap = float(a[3])
    FFT=float(a[5])
    alpha = float(a[6])
    beta = float(a[7])
    capacity_data.append(cap)
    FFT_data.append(FFT)
    alpha_data.append(alpha)
    beta_data.append(beta)
    link_data.append(link)
    link_data_copy.append((i+1,link))
    del link,a,b,c,cap,FFT,alpha, beta,q

link_data_ordered = OrderedDict(link_data_copy)
link_data_ordered = {v: k for k, v in link_data_ordered.items()}
del i, net_data, link_data_copy

G.add_edges_from(link_data)

##################Section 1.5############################
#This block reads in the flow weights in the form of num_link*num_link csv.
#Each row should be normalized to sum to 1, else it is done here

flow_weight_data = np.loadtxt(network_flow_weight_file_path, delimiter=",")
sum_of_rows = flow_weight_data.sum(axis=1)
flow_weight_data = flow_weight_data / sum_of_rows[:, np.newaxis]
del sum_of_rows

flow_weight_diagonal = [ row[i] for i,row in enumerate(flow_weight_data) ]



##################Section 1.75###########################
#This block reads in OD matrix in the zone x zone format. 
#Column 1 is the zone list, so the shape of the file zones*(zones+1)

OD_data_temp = np.loadtxt(OD_file_path, delimiter=",")
OD_matrix = OD_data_temp[:,1:]
list_zones = OD_data_temp[:,0]
list_zones = list_zones.astype(int)
list_zones = list_zones.astype(str)
list_zones=list_zones.tolist()
del OD_data_temp

##################Section 2#############################
#This block has various functions required to interface between networkx and the algorithms


def set_graph_weights(G, edge_list, weight_list):
    weights = dict(zip(edge_list, weight_list))
    nx.set_edge_attributes(G, values = weights, name = 'weight')
    
    
def get_max_dependent_link_number(G, num_links):
    length = dict(nx.all_pairs_shortest_path_length(G))
    link_adjacency_data = np.zeros((num_links, num_links))
    max_dep = 0
    for sgw in range(num_links):
        for sgw2 in range(num_links):
            if sgw==sgw2:
                link_adjacency_data[sgw][sgw2] = 0
            else:
                link_adjacency_data[sgw][sgw2] = min(length.get(link_data[sgw][0]).get(link_data[sgw2][0]),  length.get(link_data[sgw][0]).get(link_data[sgw2][1]),  length.get(link_data[sgw][1]).get(link_data[sgw2][0]), length.get(link_data[sgw][1]).get(link_data[sgw2][1])) + 1
                max_dep = max(max_dep, link_adjacency_data[sgw][sgw2])
    return link_adjacency_data, max_dep
    
def generate_symmetric_flow_weight_matrix(G, num_links, num_dependent_links, link_adjacency_data, max_num_dep_links):
    if num_dependent_links == 'all':
        num_dependent_links = max_num_dep_links
    elif num_dependent_links> max_num_dep_links:
        num_dependent_links = max_num_dep_links
    else:
        pass
    flow_weight_data = np.zeros((num_links, num_links))
    for sgw in range(num_links):
        for sgw2 in range(num_links):
            if(link_adjacency_data[sgw][sgw2] <= num_dependent_links+0.00001):
                flow_weight_data[sgw][sgw2] = random.uniform(85, 115)/100
                flow_weight_data[sgw2][sgw] = random.uniform(85, 115)/100
    sum_of_rows = flow_weight_data.sum(axis=1)
    for sgw in range(num_links):
        flow_weight_data[sgw][sgw] = sum_of_rows[sgw]
    sum_of_rows = flow_weight_data.sum(axis=1)
    flow_weight_data = flow_weight_data / sum_of_rows[:, np.newaxis]
    flow_weight_data = 0.5*flow_weight_data + 0.5*np.transpose(flow_weight_data)
    flow_weight_diagonal = [ row[i] for i,row in enumerate(flow_weight_data) ]
    return(flow_weight_data, flow_weight_diagonal)


def generate_asymmetric_flow_weight_matrix(G, num_links, num_dependent_links, link_adjacency_data, max_num_dep_links):
    # link_adjacency_data, max_num_dep_links = get_max_dependent_link_number(G, num_links)    
    if num_dependent_links == 'all':
        num_dependent_links = max_num_dep_links
    elif num_dependent_links> max_num_dep_links:
        num_dependent_links = max_num_dep_links
    else:
        pass
    flow_weight_data = np.zeros((num_links, num_links))
    for sgw in range(num_links):
        for sgw2 in range(num_links):
            if(link_adjacency_data[sgw][sgw2] <= num_dependent_links+0.00001):
                flow_weight_data[sgw][sgw2] = random.uniform(85, 115)/100
    sum_of_rows = flow_weight_data.sum(axis=1)
    for sgw in range(num_links):
        flow_weight_data[sgw][sgw] = sum_of_rows[sgw]
    sum_of_rows = flow_weight_data.sum(axis=1)
    flow_weight_data = flow_weight_data / sum_of_rows[:, np.newaxis]
    flow_weight_diagonal = [ row[i] for i,row in enumerate(flow_weight_data) ]
    return(flow_weight_data, flow_weight_diagonal)

def generate_partially_symmetric_matrix(matrix, symmetry_fraction):
    ps_matrix = (0.5+(symmetry_fraction/2))*matrix + ((1- symmetry_fraction)/2)*np.transpose(matrix)
    return ps_matrix

def cost_calculation(FFT, weighted_flow, capacity, alpha, beta):
    fft_arr = np.array(FFT)
    flows = np.array(weighted_flow)
    cap_arr = np.array(capacity)
    alpha = np.array(alpha)
    beta = np.array(beta)
    travel_times = np.multiply(fft_arr, (np.multiply(np.power((np.divide(flows,cap_arr)), beta), alpha) + np.ones(num_links)))
    #fft_arr*(1 + alpha*(flows/cap_arr)^beta)
    return travel_times.tolist()


def gradient_calculation(FFT, weighted_flow, capacity, alpha, beta):
    fft_arr = np.array(FFT)
    flows = np.array(weighted_flow)
    cap_arr = np.array(capacity)
    alpha = np.array(alpha)
    beta = np.array(beta)
    cap_beta = np.power(cap_arr, beta)
    gradient = np.divide(np.multiply(flows , np.multiply(fft_arr, np.multiply(alpha, np.multiply(beta, np.power(flows, np.subtract(beta, np.ones(len(beta)))))))), cap_beta)
    #fft_arr*(1 + alpha*(flows/cap_arr)^beta)
    return gradient.tolist()

def gradient_calculation_individual(FFT, weighted_flow, capacity, alpha, beta, index):
    fft_arr = FFT[index]
    flows = np.array(weighted_flow)
    flow_arr = flows[index]
    cap_arr = capacity[index]
    alpha_arr = alpha[index]
    beta_arr = beta[index]
    cap_beta = cap_arr**beta_arr
    gradient = fft_arr * alpha_arr*beta_arr* (flow_arr**(beta_arr-1))/ cap_beta
    gradient = np.nan_to_num(gradient)
    #fft_arr*(1 + alpha*(flows/cap_arr)^beta)
    return gradient

def list_union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def list_intersection(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def get_overlapping_segment_index(path1, path2):
    temp_arr_1 = []
    temp_arr_2 = []
    for gosi in range(len(path1)-1):
        link = (str(path1[gosi]), str(path1[gosi+1]))
        temp_arr_1.append(link_data_ordered[link] -1)
    for gosi in range(len(path2)-1):
        link = (str(path2[gosi]), str(path2[gosi+1]))
        temp_arr_2.append(link_data_ordered[link] -1)
    all_link_index = list_union(temp_arr_1, temp_arr_2)
    common_link_index = list_intersection(temp_arr_1, temp_arr_2)
    temp_arr3 = list(set(all_link_index)-set(common_link_index))
    return temp_arr_1, temp_arr_2,temp_arr3

def path_cost(G, path):
    return sum([G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)])

def GP_flow_shift_quantity(path1, path2, current_flows):
    set_graph_weights(G, link_data, cost_calculation(FFT_data, get_weighted_flows(current_flows, flow_weight_data), capacity_data, alpha_data, beta_data))
    # gradient_vector = gradient_calculation(FFT_data, get_weighted_flows(current_flows, flow_weight_data), capacity_data, alpha_data, beta_data)
    weighted_flow = get_weighted_flows(current_flows, flow_weight_data)
    basic_path_indices, nonbasic_path_indices,overlap = get_overlapping_segment_index(path1, path2)
    grad_cost = 0
    for i in range(len(overlap)):
        grad_cost+= gradient_calculation_individual(FFT_data, weighted_flow, capacity_data, alpha_data, beta_data, overlap[i])
    flow_shift = (path_cost(G, path2) - path_cost(G, path1))/grad_cost
    return flow_shift
    
    
def get_weighted_flows(flows, flow_weight_data):
    weighted_flow = np.dot(flow_weight_data, flows)
    return weighted_flow

def is_matrix_positive_definite(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
    
def get_shortest_path_flows(G, flows, zone_list):
    #load current flows on network and calculate costs
    set_graph_weights(G, link_data, cost_calculation(FFT_data, get_weighted_flows(flows, flow_weight_data), capacity_data, alpha_data, beta_data))
    # Find shortest paths from all zones
    new_flows = [0]*num_links
    for i in range(len(zone_list)):
        # print("Zone number:", zone_list[i])
        paths = nx.single_source_bellman_ford_path(G, zone_list[i], weight='weight')
        #print(type(paths))
        for j in range(len(zone_list)):
            #print(paths[str(j+1)])
            #print("Path length: ",len(paths[str(j+1)]))
            if OD_matrix[i][j] == 0:
                pass
            elif (len(paths[str(j+1)]))-1 ==0:
                #print('Zero length path')
                pass
            else:
                for k in range(len(paths[str(j+1)])-1):
                    # print(k)
                    new_flows[link_data_ordered[(paths[str(j+1)][k],paths[str(j+1)][k+1])] -1] += OD_matrix[i][j]
                    #print("This is i: ",i+1,", This is j: ", j+1, " This is path: ",paths[str(j+1)])
    return new_flows

def get_net_TT(flows):
    link_costs = cost_calculation(FFT_data, get_weighted_flows(flows, flow_weight_data), capacity_data, alpha_data, beta_data)
    TT = sum(i*j for i,j in zip(link_costs, flows))
    return TT

def get_net_SPTT(SP_flows, current_flows):
    link_costs = cost_calculation(FFT_data, get_weighted_flows(current_flows, flow_weight_data), capacity_data, alpha_data, beta_data)
    TT = sum(i*j for i,j in zip(link_costs, SP_flows))
    return TT

def get_FW_step_size(init_flows, target_flows, num_bisection_iterations):
    high = 1
    low = 0
    step_size = 0.5
    new_flow = (1-step_size)*np.array(init_flows) + (step_size)*np.array(target_flows)
    cost_fn = cost_calculation(FFT_data, get_weighted_flows(new_flow, flow_weight_data), capacity_data, alpha_data, beta_data)
    flow_diff = target_flows-init_flows
    costs = sum(x*y for x, y in list(zip(cost_fn, flow_diff)))
    for i in range(num_bisection_iterations):
        #print(costs)
        if costs > 0:
            high = step_size
            step_size = (high+low)/2 
        elif costs < 0:
            low = step_size
            step_size = (high+low)/2 
        else:
            return step_size
        new_flow = (1-step_size)*np.array(init_flows) + (step_size)*np.array(target_flows)
        cost_fn = cost_calculation(FFT_data, get_weighted_flows(new_flow, flow_weight_data), capacity_data, alpha_data, beta_data)
        flow_diff = target_flows-init_flows
        costs = sum(x*y for x, y in list(zip(cost_fn, flow_diff)))
    return step_size

def calculate_RG(flow_vector):
    TSTT = get_net_TT(flow_vector)
    target_flows = get_shortest_path_flows(G, flow_vector, list_zones)
    SPTT = get_net_SPTT(target_flows, flow_vector)
    RG = (TSTT/SPTT) -1
    return RG

def change_flow_on_path(path, flow_vector, flow_change):
    for aftp in range (len(path)-1):
        flow_vector[link_data_ordered[str(path[aftp]),str(path[aftp+1])]-1] += flow_change
    return flow_vector


class pathset:
    def __init__(self, origin, destination, list_of_paths, list_of_flows, total_demand):
        self.origin = origin
        self.destination = destination
        self.list_of_paths = list_of_paths
        self.list_of_flows = list_of_flows
        self.total_demand = total_demand

def GP_iteration(pathsets, current_flow):
    for i in pathsets:
        fixed_flows_for_iteration = copy.deepcopy(current_flow)
        origin = i.origin
        dest = i.destination
        path_list = i.list_of_paths
        flow_list = i.list_of_flows
        demand = i.total_demand
        set_graph_weights(G, link_data, cost_calculation(FFT_data, get_weighted_flows(fixed_flows_for_iteration, flow_weight_data), capacity_data, alpha_data, beta_data))
        shortest_path = nx.bellman_ford_path(G, origin, dest, weight='weight')
        
        #Check if shortest path is in used pathset
        if shortest_path in path_list:
            basic_index = path_list.index(shortest_path)
        else:
            path_list.append(shortest_path)
            if len(path_list)==1:
                flow_list.append(demand)
                change_flow_on_path(shortest_path, current_flow, demand)
            else:
                flow_list.append(0)
            basic_index = len(path_list) - 1
        
        #Shift travelers among used paths
        
        for j in range(len(path_list)):
            if j==basic_index:
                pass
            else:
                flow_shift = GP_flow_shift_quantity(path_list[basic_index], path_list[j], fixed_flows_for_iteration)
                final_flow_shift = min(max(flow_shift,0), flow_list[j])
                flow_list[basic_index] += final_flow_shift
                flow_list[j]-=final_flow_shift
                change_flow_on_path(path_list[basic_index], current_flow, final_flow_shift)
                change_flow_on_path(path_list[j], current_flow, (-1*final_flow_shift))
                
        #Calculate final flow vector
        
        #Drop unused paths
        drop_loop_length = len(flow_list)
        for j in reversed(range(drop_loop_length)):
            if flow_list[j] == 0:
                del flow_list[j]
                del path_list[j]
            else:
                pass


def MSA(num_iterations):
    step_size = 1
    init_flows = [0]*num_links
    target_flows = get_shortest_path_flows(G, init_flows, list_zones)
    new_flows = (1-step_size)*np.array(init_flows) + (step_size)*np.array(target_flows)
    for iter in range(num_iterations):
        target_flows = get_shortest_path_flows(G, new_flows, list_zones)
        step_size = 1/(iter+1)
        new_flows = (1-step_size)*np.array(new_flows) + (step_size)*np.array(target_flows)
        RG_array.append(calculate_RG(new_flows))
    print("--- %s seconds ---" % (time.time() - start_time))
    return RG_array, new_flows


def FW(num_iterations):
    num_bisection_iterations = 10
    init_flows = [0]*num_links
    target_flows = get_shortest_path_flows(G, init_flows, list_zones)
    step_size = 1
    new_flows = (1-step_size)*np.array(init_flows) + (step_size)*np.array(target_flows)
    for iter in range(num_iterations):
        target_flows = get_shortest_path_flows(G, new_flows, list_zones)
        step_size = get_FW_step_size(new_flows, target_flows, num_bisection_iterations)
        target_flows = get_shortest_path_flows(G, new_flows, list_zones)
        new_flows = (1-step_size)*np.array(new_flows) + (step_size)*np.array(target_flows)
        RG_array.append(calculate_RG(new_flows))
    print("--- %s seconds ---" % (time.time() - start_time))
    return RG_array, new_flows

def GP(num_GP_iters):
    pathsets = []

    for i in range(len(list_zones)):
        for j in range(len(list_zones)):
            if OD_matrix[i][j] == 0:
                pass
            else:
                pathsets.append(pathset(list_zones[i], list_zones[j], [], [], OD_matrix[i][j]))

    current_flows = [0]*num_links
    for GP_iters in range(num_GP_iters):
        GP_iteration(pathsets, current_flows)
        RG_value = calculate_RG(current_flows)
        RG_array.append(RG_value)
        print(current_flows)
    print(get_net_TT(current_flows))
    print("--- %s seconds ---" % (time.time() - start_time))
    return RG_array, current_flows

# ##################Section 3#############################
# #We implement MSA here


# RG_array, flow_vector = MSA(num_MSA_iterations)


# ##################Section 4#############################
# #We implement FW here



# RG_array, flow_vector = FW(num_FW_iterations)

# ##################Section 5#############################
# #We implement Gradient Projection here

RG_array, flow_vector = GP(num_GP_iterations)


# ##################Section 6#############################
# #Testing framework

# link_adjacency_data, max_dep = get_max_dependent_link_number(G, num_links)
# print(max_dep)

# for num_interacting_links in range(int(max_dep)):
#     print("Number of interacting links: ", num_interacting_links)
#     flow_weight_data, flow_weight_diagonal = generate_asymmetric_flow_weight_matrix(G, num_links, num_interacting_links, link_adjacency_data, max_dep)
#     # print("Starting MSA...")
#     # RG_array, flow_vector = MSA(num_MSA_iterations)
#     # print("Starting FW...")
#     # RG_array, flow_vector = FW(num_FW_iterations)
#     print("Starting GP...")
#     RG_array, flow_vector = GP(num_GP_iterations)
    
# np.savetxt("Asymmetric_SF_numlinkschange.csv", RG_array, delimiter=",")

# RG_array = []

# flow_weight_data_T, flow_weight_diagonal = generate_asymmetric_flow_weight_matrix(G, num_links, max_dep, link_adjacency_data, max_dep)
# for sym_frac in range(11):
#     print("Percent symmetric matrix: ", sym_frac*10)
#     flow_weight_data = generate_partially_symmetric_matrix(flow_weight_data_T, sym_frac/10)
#     # print("Starting MSA...")
#     # RG_array, flow_vector = MSA(num_MSA_iterations)
#     # print("Starting FW...")
#     # RG_array, flow_vector = FW(num_FW_iterations)
#     print("Starting GP...")
#     RG_array, flow_vector = GP(num_GP_iterations)
    
# np.savetxt("SF_sym_frac_change.csv", RG_array, delimiter=",")

