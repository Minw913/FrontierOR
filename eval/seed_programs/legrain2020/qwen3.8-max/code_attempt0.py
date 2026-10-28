import json
import argparse
import time
from gurobipy import *

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance_path', type=str, required=True)
    parser.add_argument('--solution_path', type=str, required=True)
    parser.add_argument('--time_limit', type=int, required=True)
    parser.add_argument('--log_path', type=str, default=None)
    args = parser.parse_args()
    
    # Read instance
    with open(args.instance_path) as f:
        instance = json.load(f)
    
    # Extract data
    ...
    
    # Build model
    model = Model()
    model.setParam('Seed', 0)
    model.setParam('MIPGap', 1e-4)
    model.setParam('NumericFocus', 0)
    model.setParam('Threads', 1)
    model.setParam('TimeLimit', args.time_limit)
    
    # Add variables
    ...
    
    # Add constraints
    ...
    
    # Set objective
    ...
    
    # Solve
    model.optimize()
    
    # Extract solution
    ...
    
    # Write output
    ...

if __name__ == '__main__':
    main()