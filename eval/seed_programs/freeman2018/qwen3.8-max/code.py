import json
import argparse
import time
import math
import numpy as np
from itertools import combinations

def solve(instance, time_limit):
    # Parse instance
    T = instance['parameters']['T']
    delta = instance['parameters']['delta']
    epsilon = instance['parameters']['epsilon']
    L = instance['parameters']['L']
    c = instance['parameters']['c']
    q = instance['parameters']['q']
    weekly_cycle = instance['parameters']['weekly_attraction_cycle']
    
    num_sites = instance['num_sites']
    num_customers = instance['num_customers']
    
    sites = instance['sites']
    customers = instance['customers']
    
    # Parse distances
    site_to_site = {}
    for key, val in instance['site_to_site_distances'].items():
        parts = key.split('_')
        site_to_site[(int(parts[0]), int(parts[1]))] = val
    
    cust_to_site = {}
    for key, val in instance['customer_to_site_distances'].items():
        parts = key.split('_')
        cust_to_site[(int(parts[0]), int(parts[1]))] = val
    
    # Compute attractions
    avg_biz_cust = instance['parameters']['avg_businesses_customers']
    avg_biz_sites = instance['parameters']['avg_businesses_sites']
    
    # Self-attraction for each customer
    self_attraction = {}
    for cust in customers:
        self_attraction[cust['id']] = cust['num_businesses'] / avg_biz_cust
    
    # Site attraction for each site
    site_attraction = {}
    for site in sites:
        site_attraction[site['id']] = site['num_businesses'] / avg_biz_sites
    
    # Attraction from customer i to site j in period t
    def compute_attraction(cust_id, site_id, t):
        dist = cust_to_site[(cust_id, site_id)]
        if dist == 0:
            dist = 0.001  # avoid division by zero
        weekly_factor = weekly_cycle[(t-1) % 7]
        return site_attraction[site_id] * weekly_factor / (dist ** q)
    
    # Precompute attractions
    attractions = {}
    for cust in customers:
        for site in sites:
            for t in range(1, T+1):
                attractions[(cust['id'], site['id'], t)] = compute_attraction(cust['id'], site['id'], t)
    
    # Compute attendance given event schedule
    def compute_attendance(events):
        # events is a set of (site_id, period) tuples
        attendance = {}
        for site in sites:
            for t in range(1, T+1):
                if (site['id'], t) not in events:
                    attendance[(site['id'], t)] = 0
                    continue
                
                total_att = 0
                for cust in customers:
                    numerator = attractions[(cust['id'], site['id'], t)]
                    denominator = self_attraction[cust['id']]
                    for (s2, t2) in events:
                        if abs(t2 - t) <= epsilon:
                            denominator += attractions[(cust['id'], s2, t2)]
                    proportion = min(1.0, numerator / denominator)
                    total_att += cust['population'] * proportion
                
                attendance[(site['id'], t)] = min(site['capacity'], total_att)
        
        return attendance
    
    # Compute profit given event schedule and tour
    def compute_profit(events, tour):
        attendance = compute_attendance(events)
        
        revenue = 0
        for (s, t) in events:
            site_info = next(site for site in sites if site['id'] == s)
            revenue += site_info['revenue_per_customer'] * attendance[(s, t)]
        
        fixed_costs = 0
        for (s, t) in events:
            site_info = next(site for site in sites if site['id'] == s)
            fixed_costs += site_info['fixed_cost']
        
        travel_cost = 0
        for t in range(2, T+1):
            s1 = tour[t-1]
            s2 = tour[t]
            travel_cost += c * site_to_site[(s1, s2)]
        
        # First period: distance from dummy (0) to tour[1] is 0
        # So no travel cost for first period
        
        return revenue - fixed_costs - travel_cost, attendance
    
    # ... (continue with heuristic)