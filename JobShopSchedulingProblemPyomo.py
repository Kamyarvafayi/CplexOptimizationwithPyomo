# In[]: Libraries
import pandas as pd
import pyomo.environ as pyo
import numpy as np
import itertools
from pyomo.opt import TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints
import logging
import cplex
# In[]: Initialization
# from numba import jit, cuda

np.random.seed(seed = 1)

Period_Num = 30
Unique_Product_List = ['Zipmet','Gloripa','Ezonium']
Supplied_Material = ['Mg','Fe','Au','Hg','He','Fr']
Supply_Materials_Initial_Inventory = {'Mg':50,'Fe':50,'Au':50,'Hg':50,'He':50,'Fr':50}

requiered_materials = {}
for (p,sm) in itertools.product(Unique_Product_List,Supplied_Material):
    requiered_materials[p,sm] = np.random.randint(0,6)

Recieved_Material = {}
for (sm,day) in itertools.product(Supplied_Material,range(Period_Num)):
    Recieved_Material[sm,day] = np.random.randint(0,50)

Packaging_Material = ['1','2','3','4','5','6']
Packaging_Materials_Initial_Inventory = {'1':50,'2':50,'3':50,'4':50,'5':50,'6':50}
 
requiered_Packaging_materials = {}
for (p,pm) in itertools.product(Unique_Product_List,Packaging_Material):
    requiered_Packaging_materials[p,pm] = np.random.randint(0,6)
 
Recieved_Packaging_Material = {}
for (pm,day) in itertools.product(Packaging_Material,range(Period_Num)):
    Recieved_Packaging_Material[pm,day] = np.random.randint(0,50)
 
Product_List = []
# max_Job = 10

max_Job_Ezo = 15
max_Job_Glo = 15
max_Job_Zip = 15

# for i in range(1,max_Job+1):
#     Product_List.append('Zipmet '+str(i))
#     Product_List.append('Gloripa '+str(i))
#     Product_List.append('Ezonium '+str(i))

for i in range(1,max_Job_Zip+1):
    Product_List.append('Zipmet '+str(i))

for i in range(1,max_Job_Glo+1):
    Product_List.append('Gloripa '+str(i))
 
for i in range(1,max_Job_Ezo+1):
    Product_List.append('Ezonium '+str(i))
Operations = [Op for Op in range(1,4)]
# Batches = [batch for batch in range(1,5)]
Machines = [M for M in range(1,7)]
# order = [orders for orders in range(1,5)]
days = [day for day in range(1,Period_Num+1)]

requiered_materials_per_work = {}
for (p,sm) in itertools.product(Unique_Product_List,Supplied_Material):
    for job in Product_List:
        if p in job:
            requiered_materials_per_work[job,sm] = requiered_materials[p,sm]

requiered_Packaging_materials_per_work = {}
for (p,sp) in itertools.product(Unique_Product_List,Packaging_Material):
    for job in Product_List:
        if p in job:
            requiered_Packaging_materials_per_work [job,sp] = requiered_Packaging_materials [p,sp]
# Parameters
T = [i for i in range(0,(Period_Num+1)*24,24)]
DL_Day = 24
TP = Period_Num*DL_Day
Demand = {i:1000000000000 for i in Unique_Product_List}
Shortage_Cost = {i:50000000 for i in Unique_Product_List}
# Length_Production = {i:4 for i in itertools.product(Product_List,Operations,Machines)}
# Length_Production = {i:4.5 for i in itertools.product(Product_List,Operations)}
Length_Production = {i:np.random.randint(1,2)+np.random.rand() for i in itertools.product(Product_List,Operations)}

BigM = 2*10**(4)
Batch_Size = {i:100 for i in Unique_Product_List}
WFeasibility = {i:0 for i in itertools.product(Product_List,Operations,Machines)}
for key in itertools.product(Product_List,Operations):
    if 'Gloripa' in key[0] and (key[1] == 1):
        WFeasibility[key[0],key[1], 1] = 1
        WFeasibility[key[0],key[1], 2] = 1
    elif 'Gloripa' in key[0] and (key[1] == 2):
        WFeasibility[key[0],key[1], 3] = 1
        WFeasibility[key[0],key[1], 4] = 1
    elif 'Gloripa' in key[0] and (key[1] == 3):
        WFeasibility[key[0],key[1], 6] = 1
    if 'Zipmet' in key[0] and (key[1] == 1):
        WFeasibility[key[0],key[1], 1] = 1
    elif 'Zipmet' in key[0] and (key[1] == 2):
        WFeasibility[key[0],key[1], 3] = 1
        WFeasibility[key[0],key[1], 4] = 1
        # WFeasibility[key[0],key[1], 5] = 1
    elif 'Zipmet' in key[0] and (key[1] == 3):
        WFeasibility[key[0],key[1], 5] = 1
        WFeasibility[key[0],key[1], 6] = 1
    if 'Ezonium' in key[0] and (key[1] == 1):
        WFeasibility[key[0],key[1], 2] = 1
    elif 'Ezonium' in key[0] and (key[1] == 2):
        WFeasibility[key[0],key[1], 4] = 1
    elif 'Ezonium' in key[0] and (key[1] == 3):
        WFeasibility[key[0],key[1], 5] = 1
    # random_Machine = np.random.permutation(Machines)
    # WFeasibility[key[0],key[1],random_Machine[0]] = 1
    # WFeasibility[key[0],key[1],random_Machine[1]] = 1
Warm_Up = 0.2
Warm_Up2 = 0.1
Warm_Up_Default = {i:0 for i in itertools.product(Product_List,Operations)}

# In[]: Model Creation
model_WP = pyo.ConcreteModel()

model_WP.BigM = pyo.Param(initialize = 20000)
# Defining Sets
model_WP.jobs = pyo.Set(initialize = Product_List)
model_WP.jobsj = pyo.Set(initialize = Product_List)
model_WP.par = pyo.Set(initialize = Operations)
model_WP.parj = pyo.Set(initialize = Operations)
model_WP.M = pyo.Set(initialize = Machines)
# model_WP.periods = pyo.Set(initialize = days)
model_WP.s = pyo.Set(initialize = Supplied_Material)
model_WP.pack = pyo.Set(initialize = Packaging_Material)
model_WP.P = pyo.Set(initialize = Unique_Product_List)
model_WP.t = pyo.Set(initialize = [i for i in range(Period_Num)])
model_WP.ALL = pyo.Set(initialize = ['ALL'])
model_WP.ALLplust = model_WP.ALL|model_WP.t
model_WP.JobParM = model_WP.jobs*model_WP.par*model_WP.M
model_WP.JobParMday = model_WP.jobs*model_WP.par*model_WP.M*model_WP.t

# Defining Variables
model_WP.X = pyo.Var(model_WP.jobs,model_WP.par,model_WP.jobs,model_WP.par, domain = pyo.Binary, initialize = 0)
# model_WP.I = pyo.Var(model_WP.P,model_WP.day,domain = pyo.NonNegativeIntegers, initialize = 0)
# model_WP.IS = pyo.Var(model_WP.s,model_WP.day, domain = pyo.NonNegativeIntegers, initialize = 0)
model_WP.B = pyo.Var(model_WP.jobs,model_WP.par, model_WP.M,domain = pyo.Binary, initialize = 0)
model_WP.Start_Time = pyo.Var(model_WP.jobs,model_WP.par, domain = pyo.NonNegativeReals, bounds = (0.0,BigM),initialize = 0)
model_WP.Finish_Time = pyo.Var(model_WP.jobs,model_WP.par, domain = pyo.NonNegativeReals, bounds = (0.0,BigM),initialize = 0)
# model_WP.First_Job = pyo.Var(model_WP.jobs,model_WP.par, model_WP.M,domain = pyo.Binary, initialize = 0)
model_WP.Starting_Period = pyo.Var(model_WP.jobs,model_WP.par, model_WP.t, domain = pyo.Binary, initialize = 0)
model_WP.Working_Period = pyo.Var(model_WP.jobs,model_WP.par, model_WP.t, domain = pyo.Binary, initialize = 0)
model_WP.Produced_P = pyo.Var(model_WP.P, model_WP.ALLplust, domain = pyo.NonNegativeReals,  bounds = (0,BigM), initialize = 0)
model_WP.Makespan = pyo.Var(domain = pyo.NonNegativeReals, initialize = 1000)

# model_WP.Z_Obj = pyo.Var(model_WP.P, model_WP.day,domain = pyo.NonNegativeIntegers, initialize = 0)

# Inventory Variables
model_WP.IIs = pyo.Var(model_WP.s, model_WP.t, domain = pyo.NonNegativeReals,bounds = (0,BigM))
model_WP.IIpack = pyo.Var(model_WP.pack, model_WP.t, domain = pyo.NonNegativeReals,bounds = (0,BigM))
# In[]: Time Constriants
# Constraints
def rule_c2 (model_WP, i, op):
    return (model_WP.Start_Time[i,op] + Length_Production[i,op] == model_WP.Finish_Time[i,op])
model_WP.c2 = pyo.Constraint(model_WP.jobs,model_WP.par, rule = rule_c2)

# def rule_c66 (model_WP, i, opi, j, opj, M):
#     return (model_WP.Start_Time[i,opi] <= model_WP.Finish_Time[j,opj]+2*(BigM)*(3-model_WP.X[i,opi,j,opj]-model_WP.B[i,opi,M]-model_WP.B[j,opj,M]))
# model_WP.c66 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobs,model_WP.par, model_WP.M, rule = rule_c66)
 
# def rule_c5 (model_WP, i, opi, j, opj, M):
#     if i<j:
#         return (model_WP.Start_Time[i,opi] >=Warm_Up + model_WP.Finish_Time[j,opj]-(model_WP.BigM)*(1-model_WP.X[i,opi,j,opj]))
#     else:
#         return pyo.Constraint.Skip
# model_WP.c5 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobsj,model_WP.parj, model_WP.M, rule = rule_c5)
 
# def rule_c51 (model_WP, i, opi, j, opj, M):
#     return (model_WP.Start_Time[i,opi] >= model_WP.Start_Time[j,opj]-2*(BigM)*(1-model_WP.X[i,opi,j,opj,M]))
# model_WP.c51 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobs,model_WP.par, model_WP.M, rule = rule_c51)
 
def rule_c55 (model_WP, i, opi, j, opj, M):
    if i<j:
        return (model_WP.Start_Time[i,opi] >= Warm_Up + model_WP.Finish_Time[j,opj]-(model_WP.BigM)*(3-model_WP.X[i,opi,j,opj]-model_WP.B[i,opi,M]-model_WP.B[j,opj,M]))
    else:
        return pyo.Constraint.Skip
model_WP.c55 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobsj,model_WP.parj, model_WP.M, rule = rule_c55)

def rule_c52 (model_WP, i, opi, j, opj, M):
    if i<j:
        return (model_WP.Start_Time[j,opj] >= Warm_Up + model_WP.Finish_Time[i,opi]-(model_WP.BigM)*(2+model_WP.X[i,opi,j,opj]-model_WP.B[i,opi,M]-model_WP.B[j,opj,M]))
    else:
        return pyo.Constraint.Skip
model_WP.c52 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobsj,model_WP.parj, model_WP.M, rule = rule_c52)


# def rule_c56 (model_WP, i, opi, j, opj, M):
#     if i<j:
#         return (model_WP.Start_Time[i,opi] >= model_WP.Start_Time[j,opj]-(model_WP.BigM)*(3-model_WP.X[i,opi,j,opj]-model_WP.B[i,opi,M]-model_WP.B[j,opj,M]))
#     else:
#         return pyo.Constraint.Skip
# model_WP.c56 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobsj,model_WP.parj, model_WP.M, rule = rule_c56)
 
# def rule_c57 (model_WP, i, opi, j, opj, M):
#     if i<j:
#         return (model_WP.Start_Time[j,opj] >= model_WP.Start_Time[i,opi]-(model_WP.BigM)*(2+model_WP.X[i,opi,j,opj]-model_WP.B[i,opi,M]-model_WP.B[j,opj,M]))
#     else:
#         return pyo.Constraint.Skip
# model_WP.c57 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobsj,model_WP.parj, model_WP.M, rule = rule_c57)
 
# def rule_c58 (model_WP, i, opi, j, opj, M):
#     if i<j:
#         return (model_WP.Finish_Time[i,opi] >= model_WP.Finish_Time[j,opj]-(model_WP.BigM)*(3-model_WP.X[i,opi,j,opj]-model_WP.B[i,opi,M]-model_WP.B[j,opj,M]))
#     else:
#         return pyo.Constraint.Skip
# model_WP.c58 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobsj,model_WP.parj, model_WP.M, rule = rule_c58)
 
# def rule_c59 (model_WP, i, opi, j, opj, M):
#     if i<j:
#         return (model_WP.Finish_Time[j,opj] >= model_WP.Finish_Time[i,opi]-(model_WP.BigM)*(2+model_WP.X[i,opi,j,opj]-model_WP.B[i,opi,M]-model_WP.B[j,opj,M]))
#     else:
#         return pyo.Constraint.Skip
# model_WP.c59 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobsj,model_WP.parj, model_WP.M, rule = rule_c59)
 
 
# def rule_c6 (model_WP, i, opi, j, opj, M):
#     return (model_WP.Start_Time[i,opi] <= model_WP.Finish_Time[j,opj]+2*(model_WP.BigM)*(1-model_WP.X[i,opi,j,opj]))
# model_WP.c6 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobs,model_WP.par, model_WP.M, rule = rule_c6)
 
# In[]
# def rule_c7 (model_WP, i, opi, j, opj, M):
#     return (model_WP.X[i,opi,j,opj,M] <= model_WP.B[i,opi,M] )
# model_WP.c7 = pyo.Constraint(model_WP.jobs,model_WP.par, model_WP.jobs,model_WP.par,model_WP.M, rule = rule_c7)
 
 
# def rule_c9 (model_WP, i, opi, j, opj, M):
#     return (model_WP.X[i,opi,j,opj,M] <= model_WP.B[j,opj,M] )
# model_WP.c9 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.jobs,model_WP.par, model_WP.M, rule = rule_c9)
 
# def rule_c10 (model_WP, i, opi, j, opj, M):

#     return (model_WP.X[i,opi,j,opj,M] <= model_WP.B[j,opj,M] )

# model_WP.c10 = pyo.Constraint(model_WP.jobs,model_WP.par, model_WP.jobs,model_WP.par,model_WP.M, rule = rule_c10)
 
# !!!!!!!!!!!!!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def rule_c8 (model_WP, i, op):
    return (pyo.quicksum(model_WP.B[i,op,M] for M in model_WP.M) == 1)
model_WP.c8 = pyo.Constraint(model_WP.jobs,model_WP.par, rule = rule_c8)

# def rule_c8 (model_WP, i, op):
#     return (pyo.quicksum(model_WP.B[i,op,M] for M in model_WP.M) <= 1)
# model_WP.c8 = pyo.Constraint(model_WP.jobs,model_WP.par, rule = rule_c8)
 
# def rule_c9 (model_WP, i, op1,op2):
#     if op2<op1:
#         return (pyo.quicksum(model_WP.B[i,op1,M] for M in model_WP.M) <= pyo.quicksum(model_WP.B[i,op2,M] for M in model_WP.M) )
#     else:
#         return pyo.Constraint.Skip
# model_WP.c9 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.par, rule = rule_c9)
 
def rule_c11 (model_WP, i, op1, op2):
    if op1>op2:
        return (model_WP.Start_Time[i, op1]>= Warm_Up2 + model_WP.Finish_Time[i, op2])
    else:
        return pyo.Constraint.Skip
model_WP.c11 = pyo.Constraint(model_WP.jobs,model_WP.par,model_WP.parj, rule = rule_c11)
 
 
# def rule_c18 (model_WP, i, opi, j, opj):
#     if i<j or (i==j and opi!=opj):
#         return (model_WP.X[i,opi,j,opj] + model_WP.X[j,opj,i,opi] == 1 )
#     else:
#         return pyo.Constraint.Skip
# model_WP.c18 = pyo.Constraint(model_WP.jobs,model_WP.par, model_WP.jobs,model_WP.par, rule = rule_c18)
 
def rule_c19 (model_WP, i, opi):
    return (model_WP.Makespan>=model_WP.Finish_Time[i,opi])
model_WP.c19 = pyo.Constraint(model_WP.jobs,model_WP.par, rule = rule_c19)
# In[]: Working Period and starting periods Constraints
# def rule_c20 (model_WP, i, op, t):
#     return model_WP.Start_Time[i,op] <= model_WP.Starting_Period[i,op,t]*T[t+1]+(1-model_WP.Starting_Period[i,op,t])*BigM
# model_WP.c20 = pyo.Constraint(model_WP.jobs,model_WP.par, model_WP.t, rule = rule_c20)
 
# def rule_c21 (model_WP, i, op, t):
#     return model_WP.Start_Time[i,op] >= model_WP.Starting_Period[i,op,t]*T[t]-(1-model_WP.Starting_Period[i,op,t])*BigM
# model_WP.c21 = pyo.Constraint(model_WP.jobs,model_WP.par, model_WP.t, rule = rule_c21)

# def rule_c22 (model_WP, i, op):
#     return (pyo.quicksum(model_WP.Starting_Period[i,op,t] for t in model_WP.t)==1)
# model_WP.c22 = pyo.Constraint(model_WP.jobs,model_WP.par, rule = rule_c22)

# def rule_c23 (model_WP, i, op, t):
#     return model_WP.Finish_Time[i,op] <= model_WP.Working_Period[i,op,t]*T[t+1]+(1-model_WP.Working_Period[i,op,t])*BigM
# model_WP.c23 = pyo.Constraint(model_WP.jobs,model_WP.par, model_WP.t, rule = rule_c23)

# def rule_c24 (model_WP, i, op, t):
#     return model_WP.Finish_Time[i,op] >= model_WP.Working_Period[i,op,t]*T[t]-(1-model_WP.Working_Period[i,op,t])*BigM
# model_WP.c24 = pyo.Constraint(model_WP.jobs,model_WP.par, model_WP.t, rule = rule_c24)

# def rule_c25 (model_WP, i, op):
#     return (pyo.quicksum(model_WP.Working_Period[i,op,t] for t in model_WP.t)==1)
# model_WP.c25 = pyo.Constraint(model_WP.jobs,model_WP.par, rule = rule_c25)

# # In[]: Inventory Constriants
# # Supply material Inventory
# def rule_c26 (model_WP, s, t):
#     if t == 0:
#         return (model_WP.IIs[s,0] == Supply_Materials_Initial_Inventory[s] + Recieved_Material[s,0] - pyo.quicksum(requiered_materials_per_work[i,s]*model_WP.Starting_Period[i,1,0] for i in model_WP.jobs))
#     else:
#         return (model_WP.IIs[s,t] == model_WP.IIs[s,t-1] + Recieved_Material[s,t] - pyo.quicksum(requiered_materials_per_work[i,s]*model_WP.Starting_Period[i,1,t] for i in model_WP.jobs))
# model_WP.c26 = pyo.Constraint(model_WP.s,model_WP.t, rule = rule_c26)
# # Packaging material Inventory
# def rule_c27 (model_WP, pack, t):
#     if t == 0:
#         return (model_WP.IIpack[pack,0] == Packaging_Materials_Initial_Inventory [pack] + Recieved_Packaging_Material[pack,0] - pyo.quicksum(requiered_Packaging_materials_per_work[i,pack]*model_WP.Starting_Period[i,3,0] for i in model_WP.jobs))
#     else:
#         return (model_WP.IIpack[pack,t] == model_WP.IIpack[pack,t-1] + Recieved_Packaging_Material[pack,t] - pyo.quicksum(requiered_Packaging_materials_per_work[i,pack]*model_WP.Starting_Period[i,3,t] for i in model_WP.jobs))
# model_WP.c27 = pyo.Constraint(model_WP.pack,model_WP.t, rule = rule_c27)
# In[]: Finding Production

# def rule_c28 (model_WP, p,t):
#     expr = (model_WP.Produced_P[p,t] == Batch_Size[p]*pyo.quicksum(model_WP.Working_Period[i,3,t] for i in model_WP.jobs if p in i))
#     if expr is None:
#         return pyo.Constraint.Skip()
#     else:
#         return expr
# model_WP.c28 = pyo.Constraint(model_WP.P,model_WP.t, rule = rule_c28)
# def rule_c29 (model_WP, p,t):
#     expr = (model_WP.Produced_P[p,'ALL'] == Batch_Size[p]*pyo.quicksum(model_WP.B[i,3,M] for (i,M) in model_WP.jobs*model_WP.M if p in i))
#     if expr is None:
#         return pyo.Constraint.Skip()
#     else:
#         return expr
# model_WP.c29 = pyo.Constraint(model_WP.P,model_WP.t, rule = rule_c29)
# In[]: Objective Function
# Defining Objective Function
# def Rule_OBJ(model_WP):
#     return (pyo.quicksum(model_WP.Z_Obj[p,day]*Shortage_Cost[p] for (p,day) in model_WP.P*model_WP.day))
# model_WP.OBj = pyo.Objective(rule = Rule_OBJ, sense=pyo.minimize)

def Rule_OBJ(model_WP):
    return (pyo.quicksum(model_WP.Start_Time[i,op] + model_WP.Finish_Time[i,op] for (i, op) in model_WP.jobs*model_WP.par))
model_WP.OBj = pyo.Objective(rule = Rule_OBJ, sense=pyo.minimize)

# def Rule_OBJ(model_WP):
#     return (model_WP.Makespan)
# model_WP.OBj = pyo.Objective(rule = Rule_OBJ, sense=pyo.minimize)

# def Rule_OBJ(model_WP):
#     return (pyo.quicksum(model_WP.Produced_P[p,'ALL'] for p in model_WP.P))
# model_WP.OBj = pyo.Objective(rule = Rule_OBJ, sense=pyo.maximize)
# In[]: Fixing some Variables
for i in model_WP.jobs*model_WP.par*model_WP.M:
    if WFeasibility[i[0],i[1],i[2]] == 0:
        model_WP.B[i[0],i[1],i[2]].fix(0)
        # model_WP.First_Job[i[0],i[1],i[2]].fix(0)
        # for j in model_WP.jobs*model_WP.par:
            # model_WP.X[i[0],i[1],j[0],j[1],i[2]].fix(0)
print(" The Weekly plan Model Has been Created in pyomo!")
# In[4]: Solving the model
# for checking the availability of solvers and for using various solvers in python
# pyo.SolverFactory('glpk').available() == True
# pyo.SolverFactory('gams').solve(model, solver = 'CONOPT', tee = True)
solver = pyo.SolverFactory('cplex')
solver.options['mipgap'] = 0.02
solver.options['timelimit'] = 300
# @jit(target_backend='cuda')
def Solve():
    result = solver.solve(model_WP,tee=True)
    result.write()
    return result
result = Solve()

if result.solver.termination_condition == TerminationCondition.infeasible:
    log_infeasible_constraints(model_WP, log_expression=True, log_variables=True)
    logging.basicConfig(filename='Infeasible reasons for Weekly Plan model.log', level=logging.INFO)
    raise ('ERROR: The Weekly Plan model is infeasible. Please, read the Infeasible reasons for Weekly Plan model.log')
# In[5]: Checking the result
Columns = ['jobs','operations','Machine','Start_Value','Finish_Value','B']
Finalresult = pd.DataFrame(columns = Columns)
# count = 0
for key in model_WP.jobs*model_WP.par*model_WP.M:
    if pyo.value(model_WP.B[key[0],key[1],key[2]])>=0.9:
    # count += 1
        new_Row = {'jobs':key[0],'operations':key[1],'Machine':key[2],'Start_Value':pyo.value(model_WP.Start_Time[key[0],key[1]]),'Finish_Value':pyo.value(model_WP.Finish_Time[key[0],key[1]]), 'B':pyo.value(model_WP.B[key[0],key[1],key[2]]),}
        Finalresult = pd.concat([Finalresult, pd.DataFrame([new_Row])], ignore_index = True)
    # print(count)
Finalresult.sort_values(['jobs','operations','Machine'], ascending=[True,True,True],inplace = True)

# In[] Checking X
# Columns2 = ['jobsi','operationsi','jobsj','operationsj','ValueX[i,j]','ValueX[j,i]']
# Finalresult2 = pd.DataFrame(columns = Columns2)
# # count = 0
# for key in model_WP.jobs*model_WP.par*model_WP.jobs*model_WP.par:
#     # if pyo.value(model_WP.B[key[0],key[1],key[2]])>0:
#     # count += 1
#     new_Row = {'jobsi':key[0],'operationsi':key[1],'jobsj':key[2],'operationsj':key[3], 'ValueX[i,j]':pyo.value(model_WP.X[key[0],key[1],key[2],key[3]]),'ValueX[j,i]':pyo.value(model_WP.X[key[2],key[3],key[0],key[1]])}
#     Finalresult2 = Finalresult2.append(new_Row, ignore_index = True)
#     # print(count)
# Finalresult.sort_values(['jobs','operations','Machine'], ascending=[True,True,True],inplace = True)
# In[]: Checking Starting and working Period
Columns = ['jobs','operations','period','Start_Value','Finish_Value','Starting Period','Working Period']
Finalresult3 = pd.DataFrame(columns = Columns)
for key in model_WP.jobs*model_WP.par*model_WP.t:
    if pyo.value(model_WP.Working_Period[key[0],key[1],key[2]])>=0.9 and pyo.value(model_WP.Starting_Period[key[0],key[1],key[2]])>=0.9:
        new_Row = {'jobs':key[0],'operations':key[1],'period':key[2],'Start_Value':pyo.value(model_WP.Start_Time[key[0],key[1]]),'Finish_Value':pyo.value(model_WP.Finish_Time[key[0],key[1]]),'Starting Period':key[2], 'Working Period':key[2]}
        Finalresult3 = Finalresult3.append(new_Row, ignore_index = True)
    elif pyo.value(model_WP.Working_Period[key[0],key[1],key[2]])>=0.9:
        new_Row = {'jobs':key[0],'operations':key[1],'period':key[2],'Start_Value':pyo.value(model_WP.Start_Time[key[0],key[1]]),'Finish_Value':pyo.value(model_WP.Finish_Time[key[0],key[1]]),'Starting Period':key[2]-1, 'Working Period':key[2]}
        Finalresult3 = pd.concat([Finalresult3, pd.DataFrame([new_Row])], ignore_index = True)
Finalresult3.sort_values(['jobs','operations','period'], ascending=[False,True,True],inplace = True)
# In[]: Checking End Time
Columns = ['Product','t','production']
Finalresult4 = pd.DataFrame(columns = Columns)
for key in model_WP.P*model_WP.t:
    if pyo.value(model_WP.Produced_P[key[0],key[1]]) > 0:
        new_Row = {'Product':key[0],'t':key[1],'production':pyo.value(model_WP.Produced_P[key[0],key[1]])}
        Finalresult4 = pd.concat([Finalresult4, pd.DataFrame([new_Row])], ignore_index = True)
Finalresult4.sort_values(['Product','t'], ascending=[False,True],inplace = True)
# In[]:
import matplotlib.pyplot as plt
Colors2 = {}
for i in set(Finalresult.jobs):
    Colors2[i] = [max(np.random.rand(),0.5), min(np.random.rand(),0.5), np.random.rand()]
Colors = {'1':'cyan','2':'orange', '3': 'deeppink','4':'lightblue', '5':'yellow', '6':'limegreen', '7':[0.75,0.75,0.75]}
plt.figure(figsize=(25,10))
width = 0.1
counter = 0.5
textwidth = 0.3/7
Fontsize = 8
# plt.subplot(121)
for index in Finalresult.index:
    plt.fill([Finalresult.loc[index, 'Start_Value'],Finalresult.loc[index, 'Start_Value'],Finalresult.loc[index, 'Finish_Value'],Finalresult.loc[index, 'Finish_Value']],
                      [Finalresult.loc[index, 'Machine']-counter,Finalresult.loc[index, 'Machine']+counter,Finalresult.loc[index, 'Machine']+counter,Finalresult.loc[index, 'Machine']-counter],
                      # c = Colors[str(Finalresult.loc[index, 'Machine'])],edgecolor = 'k')
                      c = Colors2[str(Finalresult.loc[index, 'jobs'])],edgecolor = 'k')
    plt.text(Finalresult.loc[index, 'Start_Value']+0.05,int(Finalresult.loc[index, 'Machine']) -counter +0.4,str(str(Finalresult.loc[index, 'jobs'])[0] + str(Finalresult.loc[index, 'jobs'])[-1] + ' Op: '+ str(Finalresult.loc[index, 'operations'])),color=[0,0,0], fontsize = Fontsize)
# for key in list(Vehicle_Code.keys()):
#         plt.text(-0.5,Vehicle_Code[key]-width,key)
for i in range(Period_Num):
    plt.axvline(x=24*(i+1), ymin = 0, ymax = 7, color ='yellow',linestyle = '-',  linewidth = 1.5)
for i in range(6):
    plt.axhline(y= i+0.5, xmin = 0, xmax = pyo.value(model_WP.Makespan), color ='k',linestyle = '-')
plt.xticks([i for i in range(0,24*Period_Num,24)])
# plt.yticks(list(Vehicle_Code.values()))
# plt.yticks([])
plt.xlim([0,pyo.value(model_WP.Makespan)+1])

plt.ylim([0.5, 6.5])

# plt.title("First Period Plan for weekly version: " + str(weekly_version) + ' and subversion: '+ str(sub_version))
plt.ylabel("Machine")
plt.xlabel("Time (hour)")
