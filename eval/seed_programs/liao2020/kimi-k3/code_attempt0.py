import argparse, json, time, math, random
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

# ---------- data ----------
load instance -> P, L, patients arrays, (c_w, c_g, c_o), cost_name

# ---------- core: f evaluation for a vertex scenario ----------
def scenario_cost(σ, t, qd, u):  # qd: list per position (d), u per position...