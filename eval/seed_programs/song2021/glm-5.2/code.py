import gurobipy as gp
from gurobipy import GRB

class MyCallback:
    def __call__(self, model, where):
        if where == GRB.Callback.MIP:
            objbnd = model.cbGet(GRB.Callback.MIP_OBJBST)
            objbst = model.cbGet(GRB.Callback.MIP_OBJBND)
            print(f"Best objective: {objbnd}, Best bound: {objbst}")

def main():
    try:
        m = gp.Model("callback_example")
        
        x = m.addVars(10, vtype=GRB.BINARY, name="x")
        m.setObjective(x.sum(), GRB.MAXIMIZE)
        m.addConstr(x.sum() <= 7, name="constr")
        
        cb = MyCallback()
        m.optimize(cb)
        
        print(f"Optimization status: {m.status}")
        if m.status == GRB.OPTIMAL:
            print(f"Optimal objective: {m.objVal}")
            for i in range(10):
                if x[i].x > 0.5:
                    print(f"x[{i}] = 1")
    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
    except AttributeError as e:
        print(f"Encountered an attribute error: {e}")

if __name__ == "__main__":
    main()