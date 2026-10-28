import argparse, json, time, math, random, sys

def load_instance(path): ...

class Instance: parse fields.

def build_initial(inst): NN tour.
def two_opt(route, tt): improvement.

Solution class:
  __init__(inst)
  set_route(route), assignDrone(j,k,v), etc.
  recompute all from scratch (travel, loads, waits) — O(nk + m).
  copy().

Local search functions operate on Solution.

evaluate full solution: compute objective from scratch.

def local_search(sol, deadline): loop improvements.

def lns(inst, sol, deadline, logger, best_ref): ...

def solve_mip(inst, warm_sol, deadline, logger): returns best solution dict/objective.