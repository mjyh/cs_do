#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import numpy as np
import sys
import gc
import scipy as sp 

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calc_distance_mat(node_count, points):
    #distance_mat = np.zeros([node_count, node_count])
    #for row in range(node_count):
    #    distance_mat[row, :] = list(map(lambda p: length(p, points[row]), points))
    distance_mat = sp.spatial.distance.squareform(sp.spatial.distance.pdist(points))
    
    return distance_mat
    
def find_next_node_greedy(node_count, points, solution):
    
    all_points = np.arange(node_count)
    current_node = solution[-1]
    
    #distances_from_current = distance_mat[current_node, :]
    distances_from_current = sp.spatial.distance.cdist([points[current_node]],points)[0]
    
    paired_distances = np.array([all_points, distances_from_current])

    available_points = np.setdiff1d( all_points, solution )
    available_pairs = paired_distances[:, available_points]    
    available_pairs = available_pairs[:, np.lexsort(available_pairs)]        
    
    next_node = np.int(available_pairs[0][0])
    
    return next_node

def greedyTSP(node_count, points):
    start_node = 0
    solution = np.array([start_node], np.int)
    for ind in range(node_count-1):
        if not ind%1000:
            print(sys.stderr, "iteration %s" % ind)
        next_node = find_next_node_greedy(node_count, points, solution)
        solution = np.append(solution, next_node)
        
    return solution

def solve_it(input_data):
    gc.collect()
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = np.ndarray([node_count, 2])
    for i in range(1, node_count+1):
        if not i%1000:
            print("iteration %s" % i)
        line = lines[i]
        parts = line.split()
        points[i-1][0] = parts[0] 
        points[i-1][1] = parts[1]

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    # solution = range(0, nodeCount)
    
    # set up distance matrix
    # distance_mat = calc_distance_mat(node_count, points)

    # calculate solution
    solution = greedyTSP(node_count, points)
    
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, node_count-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

if __name__ == '__main__':
    p1 = r'./data/tsp_51_1' 
    p2 = r'./data/tsp_100_3'
    p3 = r'./data/tsp_200_2'
    p4 = r'./data/tsp_574_1'
    p5 = r'./data/tsp_1889_1'
    p6 = r'./data/tsp_33810_1'
    all_problems = [p1, p2, p3, p4, p5, p6]
    target_problems = [p6]
    #target_problems = all_problems
    for file_location in target_problems:
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
    print(solve_it(input_data))
#    else:
#        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')