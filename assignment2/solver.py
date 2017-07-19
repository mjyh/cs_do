#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import time
import numpy as np

Item = namedtuple("Item", ['value', 'weight'])

# returns [optimum value, optimum collection] for a knapsack of capacity and
# items at itemNum or greater
def bottomUpDP(capacity, numItems, items):
    oldVals = np.zeros(capacity+1)
    newVals = np.zeros(capacity+1)
    capacities = np.arange(0, capacity+1)
    
    paths = np.full([numItems, capacity+1], True, dtype=bool)
    
    for j in range(numItems):
        if j%100 == 0:
            print("at item %s" % j)

        paths[j] = (capacities >= items[j][1]) & ((np.roll(oldVals, items[j][1]) + items[j][0]) > oldVals)
        newVals = np.where( paths[j], np.roll(oldVals, items[j][1]) + items[j][0], oldVals )
        
        oldVals = newVals
    
    optValue = 0
    optPath = []
    currentCapacity = capacity
    #get optimal value and path
    for j in reversed(range(numItems)):
        if paths[j][currentCapacity]:
            optValue += items[j][0]
            optPath = [1] + optPath
            currentCapacity -= items[j][1]
        else:
            optPath = [0] + optPath
            
    return optValue, optPath

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    
    # parse the input
    lines = input_data.split('\n')
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

   # print(firstLine)
    #print(item_count)
    items = []


    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(np.int(parts[0]), np.int(parts[1])))
        
    print("%s capacity, %s items" % (capacity, item_count))
    start = time.process_time() 
    print( "starting at %s" % start)
    # top-down dynamic approach
    optValue, optPath = bottomUpDP(capacity, item_count, items)
    end = time.process_time() 
    print( "done solving at %s, %.2f secs total" % (start, end-start))
    
    # prepare the solution in the specified output format
    output_data = str(optValue)+ ' ' + str(0) + '\n'
    output_data += ' '.join(str(taken) for taken in optPath)
    return output_data

# filepath = r'D:\Coursera\Discrete Optimization\Assignment 2\data\ks_4_0'
def runSolver(filepath):
    with open(filepath, 'r') as input_data_file:
        input_data = input_data_file.read()
    print(solve_it(input_data))
    
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        #file_location = sys.argv[1].strip()
        file_location = r'D:\Coursera\Discrete Optimization\Assignment 2\data\ks_4_0'
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

