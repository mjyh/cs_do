#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma
import warnings

# set search order
#edge_counts = edge_mat.sum(axis=0)

#edge_counts_paired = np.stack([ np.arange(0, node_count), edge_counts ])
#temp_copy = edge_counts_paired.copy()
#temp_copy[1,:] = -temp_copy[1,:]

#edge_counts_paired = edge_counts_paired[:, np.lexsort(temp_copy)]

class GreedySolver:
    def __init__(self, node_count, edge_list):
        self.node_count = node_count
        self.edge_list = edge_list
        self.reset()
    
    def reset(self):
        self.nodes_colored = np.zeros(0, dtype=np.int)
        self.colors_used = np.zeros(0, dtype=np.int)
        self.solution = np.full([self.node_count], -1, dtype=np.int)
        
        # build edge matrix
        self.edge_mat = np.zeros([self.node_count, self.node_count], dtype=np.int)
        
        for edge_pair in self.edge_list:
            self.edge_mat[edge_pair[0], edge_pair[1]] = 1
            self.edge_mat[edge_pair[1], edge_pair[0]] = 1
        
        # count total constraints
        self.total_cons = self.edge_mat.sum(axis=0)
        
        # initialize constraint matrix
        self.con_mat = np.full([self.node_count, self.node_count], np.nan)
    
    def find_next_node_most_initial_constraints(self):
        num_nodes_colored = len(self.nodes_colored)
        
        edge_counts = self.edge_mat.sum(axis=0)

        edge_counts_paired = np.stack([ np.arange(0, self.node_count), edge_counts ])
        temp_copy = edge_counts_paired.copy()
        temp_copy[1,:] = -temp_copy[1,:]
        
        edge_counts_paired = edge_counts_paired[:, np.lexsort(temp_copy)]
        
        node = edge_counts_paired[0,num_nodes_colored]
        
        return node
        
    def find_next_node_most_constrained(self):
        num_constraints = np.count_nonzero(~np.isnan(self.con_mat), axis=0) - self.total_cons
        num_constraints_paired = np.stack([ np.arange(0, self.node_count), num_constraints ])
        uncolored_nodes = np.setdiff1d( np.arange(0, self.node_count, dtype=np.int), self.nodes_colored )
        available_num_constraints_paired = num_constraints_paired[:, uncolored_nodes]
        available_most_constrained = available_num_constraints_paired[:, np.lexsort(available_num_constraints_paired)]
        node = available_most_constrained[0][0]
            
        return node
    
    def available_colors(self, node):
        return np.setdiff1d(self.colors_used, self.con_mat[node])
    
    def choose_color_min_value(self, node):
        available_colors = self.available_colors(node)
        if len(available_colors) == 0:
            color = len(self.colors_used)
            self.colors_used = np.append(self.colors_used, color) 
        else:
            color = np.nanmin(available_colors)
            
        return color

    def choose_color_random(self, node):
        available_colors = self.available_colors(node)
        if len(available_colors) == 0:
            color = len(self.colors_used)
            self.colors_used = np.append(self.colors_used, color) 
        else:
            color = np.random.choice(available_colors)
            
        return color
    
    def solve(self):
        self.reset()
        
        # assign number to solution
        for ind in range(self.node_count):
            # find node
            node = self.find_next_node_most_initial_constraints()
            # node = self.find_next_node_most_constrained()
            
            self.nodes_colored = np.append(self.nodes_colored, node)
            
            # find color
            # color = self.choose_color_min_value(node)
            color = self.choose_color_random(node)
             
            # set color and update constraints
            self.con_mat[node, self.edge_mat[node]==1] = color
            self.con_mat[self.edge_mat[node]==1, node] = color
            
            self.solution[node] = color
        
        return self.solution

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
    solver = GreedySolver(node_count, edge_list)
    solution = solver.solve()

    # prepare the solution in the specified output format
    output_data = str(solution.max()+1) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    
    return output_data

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 0:
        #file_location = sys.argv[1].strip()
        p1 = r'./data/gc_50_3'
        p2 = r'./data/gc_70_7'
        p3 = r'./data/gc_100_5'
        p4 = r'./data/gc_250_9'
        p5 = r'./data/gc_500_1'
        p6 = r'./data/gc_1000_5'

        for file_location in [p1, p2, p3, p4, p5, p6]:
            #file_location = p1
            
            with open(file_location, 'r') as input_data_file:
                input_data = input_data_file.read()
            
            print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

