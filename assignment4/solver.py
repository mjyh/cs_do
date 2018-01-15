#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import numpy as np
import sys
import gc
import scipy as sp 
import time
import itertools

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calc_solution_length(node_count, points, solution):
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, node_count-1):
        obj += length(points[solution[index]], points[solution[index+1]])
    
    return obj
    
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

def twoOpt(node_count, points, solution, start_time, min_change, time_limit):

    iterations = 0
    for ind_a in range(node_count-2):
        p_1 = solution[ind_a]
        p_2 = solution[ind_a + 1]
        d_12 = length(points[p_1], points[p_2])
        if d_12 < min_change:
            continue
        for ind_b in range(ind_a+2, node_count):
            p_3 = solution[ind_b]
            p_4 = solution[(ind_b + 1)%node_count]
            d_34 = length(points[p_3], points[p_4])
            d_13 = length(points[p_1], points[p_3])
            d_24 = length(points[p_2], points[p_4])
            if d_12 + d_34 > d_13 + d_24:
                solution[ind_a + 1] = p_3
                solution[ind_b] = p_2
                solution[ind_a + 2:ind_b] = np.flip(solution[ind_a + 2:ind_b], axis=0)
                
                
                improvement = d_12 + d_34 - d_13 - d_24
                initial_path_length = calc_solution_length(node_count, points, solution)
                #print("found cross at %s, %s | new path length: %.2f | improved by %.2f" % (ind_a, ind_b, initial_path_length, improvement))
                
                return True
            iterations += 1
            # check if we should stop
            if iterations%10000 == 0:
                #print("checking %s" % iterations)
                current_time = time.time()
                if current_time - start_time > time_limit:
                    print("breaking after running too long")
                    return False

    return False

def threeOpt(node_count, points, solution, start_time, time_limit):
    iterations = 0
    for permutation in itertools.combinations(np.arange(node_count), 3):
        iterations += 1
        # check if we should stop
        if iterations%10000 == 0:
            #print("checking %s" % iterations)
            current_time = time.time()
            if current_time - start_time > time_limit:
                print("breaking after running too long")
                return False, solution
                
        ind_a1, ind_b1, ind_c1 = permutation[0], permutation[1], permutation[2]
        ind_a2, ind_b2, ind_c2 = ind_a1+1, ind_b1+1, (ind_c1+1)%node_count
        if ind_a2 == ind_b1 or ind_b2 == ind_c1 or ind_c2 == ind_a1 or \
            ind_b1 - ind_a2 == 1 or ind_c1 - ind_b2 == 1 or (ind_a1-1)%node_count == ind_c2:
            continue
        p_a1 = solution[ind_a1] 
        p_a2 = solution[ind_a2] 
        p_b1 = solution[ind_b1] 
        p_b2 = solution[ind_b2] 
        p_c1 = solution[ind_c1] 
        p_c2 = solution[ind_c2]
  
        d_a1a2 = length(points[p_a1], points[p_a2])
        d_b1b2 = length(points[p_b1], points[p_b2])
        d_c1c2 = length(points[p_c1], points[p_c2])
        
        d_a1b1 = length(points[p_a1], points[p_b1])
        d_b1c1 = length(points[p_b1], points[p_c1])
        d_c1a1 = length(points[p_c1], points[p_a1])
        
        d_a2b2 = length(points[p_a2], points[p_b2])
        d_b2c2 = length(points[p_b2], points[p_c2])
        d_c2a2 = length(points[p_c2], points[p_a2])
        
        d_a1b2 = length(points[p_a1], points[p_b2])
        d_b1c2 = length(points[p_b1], points[p_c2])
        d_c1a2 = length(points[p_c1], points[p_a2])
    
        comb_dists = [ d_a1a2 + d_b1b2 + d_c1c2,
                      d_c1a1 + d_b1c2 + d_a2b2,
                      d_a1b1 + d_c1a2 + d_b2c2,
                      d_b1c1 + d_a1b2 + d_c2a2,
                      d_a1b2 + d_c1a2 + d_b1c2, ]
        
        min_comb = np.argmin(comb_dists)
        
        if min_comb == 0:
            continue
        else:
            initial_path_length = calc_solution_length(node_count, points, solution)
            improvement = comb_dists[0] - comb_dists[min_comb]
            #print("found cross at %s, %s, %s | %s | new path length: %.2f | improved by %.2f" % (ind_a1, ind_b1, ind_c1, min_comb, initial_path_length, improvement))
            if min_comb == 1:
                if ind_c2 == 0:
                    solution = np.concatenate([solution[:ind_a1+1], np.flip(solution[ind_b2:ind_c1+1], axis=0), solution[ind_a2:ind_b1+1]])
                else:
                    solution = np.concatenate([solution[:ind_a1+1], np.flip(solution[ind_b2:ind_c1+1], axis=0), solution[ind_a2:ind_b1+1], solution[ind_c2:]])

            elif min_comb == 2:
                if ind_c2 == 0:
                    solution = np.concatenate([solution[:ind_a1+1], np.flip(solution[ind_a2:ind_b1+1], axis=0), np.flip(solution[ind_b2:ind_c1+1], axis=0)])
                else:
                    solution = np.concatenate([solution[:ind_a1+1], np.flip(solution[ind_a2:ind_b1+1], axis=0), np.flip(solution[ind_b2:ind_c1+1], axis=0), solution[ind_c2:]])

            elif min_comb == 3:
                if ind_c2 == 0:
                    solution = np.concatenate([solution[:ind_a1+1], solution[ind_b2:ind_c1+1], np.flip(solution[ind_a2:ind_b1+1], axis=0)])
                else:
                    solution = np.concatenate([solution[:ind_a1+1], solution[ind_b2:ind_c1+1], np.flip(solution[ind_a2:ind_b1+1], axis=0), solution[ind_c2:]])

            elif min_comb == 4:
                if ind_c2 == 0:
                    solution = np.concatenate([solution[:ind_a1+1], solution[ind_b2:ind_c1+1], solution[ind_a2:ind_b1+1]])
                else:
                    solution = np.concatenate([solution[:ind_a1+1], solution[ind_b2:ind_c1+1], solution[ind_a2:ind_b1+1], solution[ind_c2:]])

            return True, solution

    return False, solution

def greedy_tsp_given_start(node_count, points, time_limit, start_node):
    # find an initial solution
    solution = np.array([start_node], np.int)
    print("trying start node %s" % start_node)
    for ind in range(node_count-1):
        #if not ind%5000:
        #    print("initial_solution: iteration %s" % ind)
        next_node = find_next_node_greedy(node_count, points, solution)
        solution = np.append(solution, next_node)
    
    initial_path_length = calc_solution_length(node_count, points, solution)
    #print("initial path length: %.2f" % initial_path_length)
    
    # perform two opt
    had_changes = True
    start_time = time.time()
    while had_changes:
        had_changes = twoOpt(node_count, points, solution, start_time, min_change, time_limit)
            
    # perform two opt
    had_changes = True
    start_time = time.time()
    while had_changes:
        had_changes, solution = threeOpt(node_count, points, solution, start_time, time_limit)
    
    return solution

def greedyTSP_all_starts(node_count, points, time_limit):
    best_solution = None
    best_length = None
    for start_node in np.arange(node_count):
        new_solution = greedy_tsp_given_start(node_count, points, time_limit, start_node)
        new_length = calc_solution_length(node_count, points, new_solution)
        if best_solution is None or new_length < best_length:
            best_solution = new_solution
            best_length = new_length
            print("new best solution | start node %s | length %s" % (start_node, best_length))
    
    return best_solution

def greedyTSPManager(node_count, points):
    
    # set meta-parameters
    time_limit = 120
    min_change = 0.0
    start_node = 1
    if node_count == 51:
        problem = 'p1'
    elif node_count == 100:
        problem = 'p2'
    elif node_count == 200:
        problem = 'p3'
    elif node_count == 574:
        problem = 'p4'
    elif node_count == 1889:
        problem = 'p5'
    else:
        problem = 'p6'
        start_node = 0
        time_limit = 120
        min_change = 5000
    
    print("solving %s" % problem)
    
    if problem in ['p1', 'p2']:
        best_solution = None
        for start_node in range(node_count):
            new_solution = solve_with_start_node(node_count, points, start_node, time_limit, min_change)
            if best_solution is None:
                best_solution = new_solution
            if calc_solution_length(node_count, points, new_solution) < calc_solution_length(node_count, points, best_solution):
                best_solution = new_solution
        best_solution = best_solution
    else:
        best_solution = solve_with_start_node(node_count, points, start_node, time_limit, min_change)
    
    return best_solution

def solve_it(input_data):
    gc.collect()
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = np.ndarray([node_count, 2])
    for i in range(1, node_count+1):
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
    solution = greedyTSPManager(node_count, points)
    
    # calculate the length of the tour
    obj = calc_solution_length(node_count, points, solution)

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
    target_problems = [p2]
    #target_problems = all_problems
    for file_location in target_problems:
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
    print(solve_it(input_data))
#    else:
#        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')