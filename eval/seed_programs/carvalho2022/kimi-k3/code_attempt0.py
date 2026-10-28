def callback(model, where):
    if where == GRB.Callback.MIPSOL:
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        # build solution dict
        ...
        logger.log_solution(obj, sol)