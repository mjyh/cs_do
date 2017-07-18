#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import time

Item = namedtuple("Item", ['index', 'value', 'weight'])
item_count = None
items = None
maxValue = 0
maxPath = []
cache = {}


# returns [optimum value, optimum collection] for a knapsack of capacity and
# items at itemNum or greater
def calcOpt(capacity, itemNum):
    global item_count
    global items
    global maxValue
    global maxPath
    
    #print( "calcOpt( %s, %s)" % (capacity, itemNum))
    
    #print("item_count: %s" % item_count)
    if itemNum == item_count - 1 or capacity <= 0:
        #print ("calcOpt( %s, %s) returning [%s, %s] " % (capacity, itemNum, 0, [0] * (item_count - itemNum)))
        return [0, [0] * (item_count - itemNum)]

    # get values with choosing item
    if capacity >= items[itemNum][2]:
        withItemVal, withItemPath = calcOpt(capacity - items[itemNum][2], itemNum+1)
        withItemVal += items[itemNum][1]
    else:
        withItemVal = -1
        withItemPath = None
    
    # get values without choosing item
    withoutItemVal, withoutItemPath = calcOpt(capacity, itemNum+1)
    
    if withItemVal > withoutItemVal:
        betterValue = withItemVal
        betterPath = [1] + withItemPath
    else:
        betterValue = withoutItemVal
        betterPath = [0] + withoutItemPath
    
    if betterValue > maxValue:
        maxValue = betterValue
        maxPath = betterPath
    
    #print ("calcOpt( %s, %s) returning [%s, %s] " % (capacity, itemNum, betterValue, betterPath))
    return betterValue, betterPath

def solve_it(input_data):
    # Modify this code to run your optimization algorithm
    global item_count
    global items
    global maxValue
    global maxPath
    
    # parse the input
    lines = input_data.split('\n')
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

   # print(firstLine)
    #print(item_count)
    items = []
    maxValue = 0
    maxPath = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
        
    print("%s items" % item_count)
    start = time.process_time() 
    print( "starting at %s" % start)
    # top-down dynamic approach
    optValue, optPath = calcOpt(capacity, 0)
    end = time.process_time() 
    print( "done solving at %s, %.2f secs total" % (start, end-start))
    
    # in case we didn't pick the first couple items
    if len(maxPath) < item_count:
        maxPath = [0] * (item_count - len(maxPath)) + maxPath
    
    # prepare the solution in the specified output format
    output_data = str(maxValue)+ ' ' + str(0) + '\n'
    output_data += ' '.join(str(taken) for taken in maxPath)
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

