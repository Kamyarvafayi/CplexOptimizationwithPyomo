# In[]: SetCovering
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import math
import random
Set_Covering = pyo.ConcreteModel()
np.random.seed(seed = 42)
c = 200
x = np.random.randint(0,30,c)
y = np.random.randint(0,30,c)
r = np.random.randint(10, 100, c)
f = np.random.randint(10,90,c) 
Cap = np.random.randint(500,1000,c)
De = np.random.randint(2,5,c)
w1 = 0.2
w2 = 0.8
d = np.zeros((c,c))
a = np.zeros((c,c))
for i in range(c):
    for j in range(c):
        if i == j:
            d[i,j] = np.random.uniform(0,2)
        else:
            d[i,j] = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
        if d[i,j] <= r[i]:
            a[i,j] = 1
Set_Covering.i = pyo.Set(initialize = [i for i in range(c)])
Set_Covering.j = pyo.Set(initialize = [j for j in range(c)])
Set_Covering.xij = pyo.Var(Set_Covering.i, Set_Covering.j, domain = pyo.Binary)
Set_Covering.yi = pyo.Var(Set_Covering.i, domain = pyo.Binary)

Set_Covering.OBJ = pyo.Objective(expr = w1*pyo.quicksum(Set_Covering.xij[i,j]*d[i,j] for (i,j) in Set_Covering.i*Set_Covering.j) + w2*pyo.quicksum(Set_Covering.yi[i]*f[i] for i in Set_Covering.i), sense = pyo.minimize)

def rule1(Set_Covering, j):
    return (pyo.quicksum(Set_Covering.xij[i,j] for i in Set_Covering.i) >= 1)
Set_Covering.Constraint1=pyo.Constraint(Set_Covering.j, rule = rule1)

def rule2(Set_Covering, i):
    return (pyo.quicksum(Set_Covering.yi[i]  for i in Set_Covering.i) <= 10)
Set_Covering.Constraint2=pyo.Constraint(Set_Covering.i, rule = rule2)

def rule3(Set_Covering, i, j):
    return (Set_Covering.xij[i,j] <= a[i,j] * Set_Covering.yi[i])
Set_Covering.Constraint3=pyo.Constraint(Set_Covering.i, Set_Covering.j, rule=rule3)

def rule4(Set_Covering, i):
    return (pyo.quicksum(Set_Covering.xij[i,j] * De[j]  for j in Set_Covering.j) <= Cap[i])
Set_Covering.Constraint4=pyo.Constraint(Set_Covering.i, rule = rule4)


solver = pyo.SolverFactory('cplex')
solver.options['mipgap'] = 0.001
result = solver.solve(Set_Covering, tee=True)
result.write()

results = np.zeros((c,c))
for i in range(c):
    print(pyo.value(Set_Covering.yi[i]))
    for j in range(c):
        results[i,j] = pyo.value(Set_Covering.xij[i,j])
print(pyo.value(Set_Covering.OBJ))

Assignments = pd.DataFrame(index=range(c), columns=range(c))
for i in range(c):
    for j in range(c):
        if pyo.value(Set_Covering.xij[i,j] < 0.1):
            Assignments.iloc[i,j] = 0
        else:
            Assignments.iloc[i,j] = 1

fig, ax = plt.subplots(figsize=(6,6))

for i in range(c):
    if pyo.value(Set_Covering.yi[i]) == 0:
        plt.plot(x[i],y[i],'ro')
    else:
       plt.plot(x[i],y[i],'y^',markersize=18)  
       cir = Circle((x[i],y[i]), radius=r[i],facecolor='none',edgecolor=(0,0.1,0.5),alpha=0.9,linestyle=':')
       ax.set_aspect('equal', adjustable='box')
       ax.add_patch(cir)
       ax.autoscale()
    for j in range(c):
        if Assignments.iloc[i,j] == 1:
            plt.plot([x[i],x[j]],[y[i],y[j]],'black',linestyle='--',markersize=9)
plt.show()
