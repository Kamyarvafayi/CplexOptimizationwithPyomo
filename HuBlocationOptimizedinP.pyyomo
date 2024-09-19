
mahyar vafaee
12:34 PM (0 minutes ago)
to me

# In[]: HUB (Single-Allocation)
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopy.distance
import folium as fol
from folium.plugins import BeautifyIcon
import io
from PIL import Image
from selenium import webdriver
import random
points = pd.read_excel(r'C:\Users\k.vafaeinejad\Desktop\Kamyar Vafaeinejad\Optimization Models\HUB\Hub Points.xlsx')
c = np.shape(points)[0]
d = np.zeros((c,c))
W = np.zeros((c,c))
np.random.seed(seed=1)
F = np.random.randint(1000000,5000000,c); F[1] = 300000
B = 5000000
for i in range(np.shape(W)[0]):
    for j in range(np.shape(W)[1]):
        if i == j:
            W[i,j] = 0
        else:
            np.random.seed(seed=1)
            W[i,j] = np.random.randint(50,300)  
W[2,:] = np.random.randint(500,1000)
W[:,2] = np.random.randint(500,1000)
W[1,:] = np.random.randint(400,600)
W[:,1] = np.random.randint(400,600);W[1,1] = 0; W[2,2] = 0
coords = pd.DataFrame(0, index = range(c), columns = ('Lat','Lon'))
for i in range(c):
    coords.iloc[i,:] = (points.iloc[i,2], points.iloc[i,1])
for i in range(c):
    for j in range(i,c):
        if i != j:
           d[i,j] = geopy.distance.geodesic(tuple(coords.iloc[i,:]),tuple(coords.iloc[j,:])).km
           d[j,i] = d[i,j]
        else:
           d[i,j] == 0
           
HUB = pyo.ConcreteModel()
# P = 3
Alpha = 0.4
HUB.i = pyo.Set(initialize = [i for i in range(0,c)])
HUB.j = pyo.Set(initialize = [j for j in range(0,c)])
HUB.k = pyo.Set(initialize = [k for k in range(0,c)])
HUB.l = pyo.Set(initialize = [l for l in range(0,c)])
HUB.xijkl = pyo.Var(HUB.i, HUB.j,HUB.k,HUB.l ,domain = pyo.NonNegativeReals)
HUB.xik = pyo.Var(HUB.i,HUB.k, domain = pyo.Binary)
HUB.Hk = pyo.Var(HUB.k, domain = pyo.Binary)


HUB.OBJ = pyo.Objective(expr = pyo.quicksum(HUB.xijkl[i,j,k,l] * W[i,j] * (d[i,k] + Alpha * d[k,l] + d[l,j]) for (i,j,k,l) in HUB.i * HUB.j * HUB.k * HUB.l))
                       
def rule1(HUB,i,j,k):
    return (pyo.quicksum(HUB.xijkl[i,j,k,l]  for l in HUB.l) == HUB.xik[i,k])
HUB.Constraint1=pyo.Constraint(HUB.i, HUB.j, HUB.k, rule = rule1)  

def rule2(HUB,i,j,l):
    return (pyo.quicksum(HUB.xijkl[i,j,k,l]  for k in HUB.k) == HUB.xik[j,l])
HUB.Constraint2=pyo.Constraint(HUB.i, HUB.j, HUB.l, rule = rule2)

def rule3(HUB,i,k):
    return (HUB.xik[i,k]  <= HUB.Hk[k])
HUB.Constraint3=pyo.Constraint(HUB.i, HUB.k, rule = rule3)

# def rule4(HUB):
#     return (pyo.quicksum(HUB.Hk[k] for k in HUB.k) == P)
# HUB.Constraint4=pyo.Constraint(rule = rule4)

def rule4(HUB):
    return (pyo.quicksum(HUB.Hk[k] * F[k] for k in HUB.k) <= B)
HUB.Constraint4=pyo.Constraint(rule = rule4)

def rule5(HUB, i):
    return (pyo.quicksum(HUB.xik[i,k] for k in HUB.k)  == 1)
HUB.Constraint5=pyo.Constraint(HUB.i, rule = rule5)                      


solver = pyo.SolverFactory('cplex')
solver.options['mipgap'] = 0.001  
solver.options['timelimit'] = 6000                
result = solver.solve(HUB, tee = True)
result.write()          
print(result)   


map = fol.Map(location=[points.iloc[1,2],points.iloc[1,1]], tiles="CartoDB Positron", zoom_start=5, prefer_canvas = True)

color = ['Olive','Orange','Red','light green','Yellow']

listicons1 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[4],
    fill_color= color[4],
)

listicons2 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[1],
    fill_color= color[1],
)

listicons3 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[3],
    fill_color= color[3],
)

listicons4 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[4],
    fill_color= color[4],
)

listicons5 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[1],
    fill_color= color[1],
)

listicons6 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[3],
    fill_color= color[3],
)

listicons7 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[4],
    fill_color= color[4],
)

listicons8 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[1],
    fill_color= color[1],
)

listicons9 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[3],
    fill_color= color[3],
)

listicons10 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[4],
    fill_color= color[4],
)

listicons11 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[1],
    fill_color= color[1],
)

listicons12 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[3],
    fill_color= color[3],
)

listicons24 = BeautifyIcon(
    icon='star',
    inner_icon_style='font-size:14px',
    border_color= color[3],
    fill_color= color[3],
)

icons_dict = {1:listicons1,2:listicons2,3:listicons3,4:listicons4,5:listicons5,
              6:listicons6,7:listicons7,8:listicons8,9:listicons9,10:listicons10,
              11:listicons11,12:listicons12,24:listicons24}


for k in range(points.shape[0]):
    if pyo.value(HUB.Hk[k]) == 0:
        fol.Circle([points.iloc[k,2],points.iloc[k,1]],radius = 13000,
                                      color = color[2],
                                      fill_color = color[2],
                                      fill_opacity = 0.8).add_child(fol.Popup(points.iloc[k,0])).add_to(map)
    else:
        fol.Marker([points.iloc[k,2],points.iloc[k,1]],popup = points.iloc[k,0],icon = icons_dict[k]).add_to(map)
       


for i in range(points.shape[0]):
    for k in range(points.shape[0]):
        if pyo.value(HUB.xik[i,k]) == 1:
            fol.PolyLine([[points.iloc[i,2],points.iloc[i,1]],[points.iloc[k,2],points.iloc[k,1]]],color = 'black', dash_array = '10').add_to(map)

HUB_Connections = pd.DataFrame(False, index = range(c), columns = ['IsHub?']) 
HUBS = []
for k in range(c):
    if pyo.value(HUB.Hk[k]) == 1:
        HUB_Connections.iloc[k,0] = True
        HUBS.append(k)
#Connecting Hubs to each other:
for k1 in range(len(HUBS)):
    for k2 in range(len(HUBS)):
        if k1 != k2:
            fol.PolyLine([[points.iloc[HUBS[k1],2],points.iloc[HUBS[k1],1]],[points.iloc[HUBS[k2],2],points.iloc[HUBS[k2],1]]],color = 'Blue', dash_array = '10').add_to(map)
        
    
        

# img_data = map._to_png(5)
# img = Image.open(io.BytesIO(img_data))
# img.save('Hubmapimage.png')
 
map.save('HUBMAP.html')
