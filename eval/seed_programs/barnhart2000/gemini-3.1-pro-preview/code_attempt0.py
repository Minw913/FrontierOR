m.Params.TimeLimit = args.time_limit
    m.Params.MIPFocus = 1
    m.optimize(cb)