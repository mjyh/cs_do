#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
import scipy as sp

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def save_lp(facility_count, facility_cost, facility_capacity, customer_count, customer_demand, distances):
    with open(r'mylp.lp', 'w') as out_file:
        ### Objective function
        out_file.write('Minimize\n')
        out_file.write('\tobj:')
        
        # write facility costs
        for facil_ind in range(facility_count):
            out_file.write(' + %f f_%i' % ( facility_cost[facil_ind], facil_ind ) )
        
        # write distance costs
        for facil_ind in range(facility_count):
            for cust_ind in range(customer_count):
                out_file.write(" + %f d_%i_%i" % ( distances[cust_ind, facil_ind], cust_ind, facil_ind ) )
        
        out_file.write('\n')
        
        ### Constraints
        out_file.write('Subject To\n')
        
        # capacity constraints
        for facil_ind in range(facility_count):
            out_file.write("\tcapacity_%i:" % facil_ind)
            for cust_ind in range(customer_count):
                out_file.write(" + %f d_%i_%i" % (customer_demand[cust_ind], cust_ind, facil_ind) )
            out_file.write(" <= %f\n" % facility_capacity[facil_ind])
                
        # each customer gets a facility
        for cust_ind in range(customer_count):
            out_file.write("\tcustomer_%i:" % cust_ind)
            for facil_ind in range(facility_count):
                out_file.write(" + d_%i_%i" % (cust_ind, facil_ind ) )
            out_file.write(" >= 1\n")
        
        # define facility helper
        for facil_ind in range(facility_count):
            out_file.write("\tfacilty_helper_%i:" % facil_ind)
            for cust_ind in range(customer_count):
                out_file.write(" + d_%i_%i" % (cust_ind, facil_ind) )
            out_file.write(" - %i f_%i <= 0\n" % ( customer_count, facil_ind ) )
        
        ### Binary
        out_file.write("Binary\n")
        for facil_ind in range(facility_count):
            out_file.write("f_%i\n" % facil_ind)
        for facil_ind in range(facility_count):
            for cust_ind in range(customer_count):
                out_file.write("d_%i_%i\n" % ( cust_ind, facil_ind ) )
        
        out_file.write('End\n')
    return

def run_optimizer():
    return

def prepare_solution():
    return

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facility_locations = np.zeros([facility_count, 2])
    facility_capacity = np.zeros([facility_count])
    facility_cost = np.zeros([facility_count])
    customer_locations = np.zeros([customer_count, 2])
    customer_demand = np.zeros([customer_count])
    
    facilities = []
    for i in range(0, facility_count):
        parts = lines[i+1].split()
        facility_cost[i] = parts[0]
        facility_capacity[i] = parts[1]
        facility_locations[i, 0] = float(parts[2])
        facility_locations[i, 1] = float(parts[3])

    customers = []
    for i in range(0, customer_count):
        line_ind = i + facility_count + 1    
        parts = lines[line_ind].split()
        customer_demand[i] = float(parts[0])
        customer_locations[i, 0] = parts[1]
        customer_locations[i, 1] = parts[2]

    distances = sp.spatial.distance.cdist(customer_locations, facility_locations)
    
    print("Num facilities: %s, num customers: %s" % (facility_count, customer_count))
    print(distances)
    
    save_lp(facility_count, facility_cost, facility_capacity, customer_count, customer_demand, distances)
    run_optimizer()
    prepare_solution()

    output_data = None
    return output_data

import sys

if __name__ == '__main__':
    file_location = r'./data/fl_3_1'
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    solve_it(input_data)

