import argparse, json, math, os, sys, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_path", required=True)
    parser.add_argument("--solution_path", required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    from solution_logger import SolutionLogger
    logger = SolutionLogger(args.log_path, sense="maximize") if args.log_path else None

    t0 = time.time()
    deadline = t0 + args.time_limit

    with open(args.instance_path) as f:
        inst = json.load(f)
    n = inst["n"]; m = inst["m"]
    capacity = inst["capacity"]
    weights = inst["weights"]
    values = inst["values"]
    pairwise = inst["pairwise_values"]

    # filter packable items
    packable = [i for i in range(n) if weights[i] <= capacity]
    ...