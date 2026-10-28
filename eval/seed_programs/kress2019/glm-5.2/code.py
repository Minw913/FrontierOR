import json
import argparse
from solution_logger import SolutionLogger

def solve(instance, time_limit):
    S = instance['S']
    n = instance['n']
    m = instance['m']
    p = instance['p']
    sigma_w = instance['sigma_w']
    sigma_l = instance['sigma_l']
    seaside_containers = instance['seaside_containers']
    landside_containers = instance['landside_containers']
    
    # Simple heuristic: seaside crane delivers directly, landside crane handles landside
    # ...