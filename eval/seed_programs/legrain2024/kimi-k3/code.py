import argparse
import json
import time
import os

def load_instance(path):
    with open(path) as f:
        return json.load(f)

def parse_day_key(key):
    # "day_{d}"
    return int(key.split("_")[1])