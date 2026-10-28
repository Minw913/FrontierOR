def solve_dp(panel_l, panel_w, items):
    # dp[l][w] = max profit from rectangle l×w (ignoring copies)
    # Also track what items are used
    
    # Collect all relevant dimensions
    lengths = set([panel_l])
    widths = set([panel_w])
    for item in items:
        lengths.add(item['length'])
        widths.add(item['width'])
    
    # Sort dimensions
    lengths = sorted(lengths)
    widths = sorted(widths)
    
    # Create DP table
    # dp[l][w] = max profit
    dp = {}
    
    for l in lengths:
        for w in widths:
            best = 0
            # Check if this rectangle matches an item
            for item in items:
                if (item['length'] == l and item['width'] == w) or \
                   (item['length'] == w and item['width'] == l):
                    best = max(best, item['profit'])
            
            # Try vertical cuts
            for x in lengths:
                if x < l and (l - x) in set(lengths):
                    if (x, w) in dp and (l - x, w) in dp:
                        best = max(best, dp[(x, w)] + dp[(l - x, w)])
            
            # Try horizontal cuts
            for y in widths:
                if y < w and (w - y) in set(widths):
                    if (l, y) in dp and (l, w - y) in dp:
                        best = max(best, dp[(l, y)] + dp[(l, w - y)])
            
            dp[(l, w)] = best
    
    return dp[(panel_l, panel_w)]