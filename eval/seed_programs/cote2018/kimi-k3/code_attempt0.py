import argparse, json, math, os, sys, time
from collections import defaultdict

def main():
    parser ...
    args = parser.parse_args()
    t0 = time.time()
    with open(args.instance_path) as f: inst = json.load(f)
    W = int(inst["bin_capacity"])
    items = inst["items"]
    # aggregate demand by width
    dem = {}
    for it in items:
        w = int(it["width"]); d = int(it["demand"])
        if d <= 0: continue
        dem[w] = dem.get(w, 0) + d
    widths = sorted(dem.keys(), reverse=True)
    ...