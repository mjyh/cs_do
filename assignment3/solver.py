#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma
import warnings


def greedy(node_count, edge_list):
   
    solution = np.full([node_count], np.nan, dtype=np.int8)

    colors_used = np.zeros(0, dtype=np.int8)
    
    # build edge matrix
    edge_mat = np.zeros([node_count, node_count], dtype=np.int8)
    
    for edge_pair in edge_list:
        edge_mat[edge_pair[0], edge_pair[1]] = 1
        edge_mat[edge_pair[1], edge_pair[0]] = 1
    
    # initialize constraint matrix
    con_mat = np.full([node_count, node_count], np.nan)
    
    # set search order
    edge_counts = edge_mat.sum(axis=0)
    
    edge_counts_paired = np.stack([ np.arange(0, node_count), edge_counts ])
    temp_copy = edge_counts_paired.copy()
    temp_copy[1,:] = -temp_copy[1,:]
    
    edge_counts_paired = edge_counts_paired[:, np.lexsort(temp_copy)]
    
    # assign number to solution
    for ind in range(node_count):
        node = edge_counts_paired[0, ind]
        
        available_colors = np.setdiff1d(colors_used, con_mat[node])
        if len(available_colors) == 0:
            color = len(colors_used)
            colors_used = np.append(colors_used, color) 
        else:
            color = np.nanmin(available_colors)

        con_mat[node, edge_mat[node]==1] = color
        con_mat[edge_mat[node]==1, node] = color
        
        solution[node] = color
        
    return solution

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    np.set_printoptions(threshold=1000, linewidth = 150)
    warnings.filterwarnings("ignore")
    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edge_list = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edge_list.append((int(parts[0]), int(parts[1])))

    # run a solver
    solution = greedy(node_count, edge_list)

    # prepare the solution in the specified output format
    output_data = str(solution.max()+1) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    
    return output_data

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        file_location = r'data//gc_20_3'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

