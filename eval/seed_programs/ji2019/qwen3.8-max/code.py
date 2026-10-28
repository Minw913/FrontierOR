import json
import argparse
import time
from gurobipy import *

def solve(instance_path, solution_path, time_limit, log_path=None):
    # Read instance
    with open(instance_path) as f:
        data = json.load(f)
    
    N = data['num_ships']
    num_berths = data['num_berths']
    W = data['chamber_width_W']
    L = data['chamber_length_L']
    Du = data['lockage_duration_Du']
    MT = data['min_time_between_lockages_MT']
    sc = data['lock_start_availability_sc']
    ek = data['berth_end_availability_ek']
    C_max = data['C_max']
    
    ships = data['ships']
    berths = data['berths']
    handling_times = data['handling_times']
    
    # ... formulate and solve