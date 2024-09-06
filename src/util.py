# #src/util

#     def get_objective(self, model, MainObjFnc, ObjFnc2=None, ObjFnc3=None, eps2=None, eps3=None, r2=None, r3=None):
#         ## Define Objective
#         if ObjFnc2 is not None:
#             model.setObjective(MainObjFnc)

#             model.addConstr(ObjFnc2 == eps2)
        
#         elif ObjFnc3 is not None:
#             model.setObjective(MainObjFnc)

#             model.addConstr(ObjFnc2 == eps2)
#             model.addConstr(ObjFnc3 == eps3)

#         elif r2 is not None and r3 is not None:
#             # Slack variables:
#             r1 = 1e-3
#             s2 = model.addVar(lb = 0, ub = 1000, vtype=GRB.CONTINUOUS, name="s2")
#             s3 = model.addVar(lb = 0, ub = 1000, vtype=GRB.CONTINUOUS, name="s3")

#             model.addConstr((ObjFnc2 + s2) == eps2)
#             model.addConstr((ObjFnc3 + s3) == eps3)

#             model.setObjective(MainObjFnc - r1 * (s2/r2 + s3/r3))
