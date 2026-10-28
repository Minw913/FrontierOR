def get_cond_var(pattern_el, n, d, x, W, R, S):
    dt, sh = pattern_el
    if dt == "Work":
        if sh == "Any":
            return W[n, d]
        else:
            return x[n, d, sh]
    elif dt == "Rest":
        return R[n, d]
    elif dt == "Any":
        return None # 1
    else:
        # fallback
        return None