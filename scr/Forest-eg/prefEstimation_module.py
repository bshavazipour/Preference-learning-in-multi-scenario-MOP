# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:45:12 2021

@author: babshava
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import csv
import seaborn as sns
import math


# Initializing the constants
#k = 4 # #of objectives
#s = 12 # #of scenarios
#m = s*k # # of meta-objectives
#q = 4  # # of scenarios for which the DM's preferences are available

def forest_prefs(k,s,m,q):
    """
    - Read the data frame including ideal and nadir values for all objectives and all scenarios,
    as well as the DM's preferences and return them in separate arrays
    
    Returns
    -------
    pref : numpy.ndarray
        Actual DM's preferences (in selesctied scenarios)
    ideal : numpy.ndarray
        Ideal values for all objectives in all scenarios
    nadir : numpy.ndarray
        Nadir values for all objectives in all scenarios
    prefi : numpy.ndarray
        Actual DM's preferences (in selesctied scenarios)

    """
    

    # Reading an Exel file including ideal and nadir values for all objectives and all scenarios in a Panda data frame
    df = pd.read_excel(r'C:\Users\babshava\SBMOP_forest\sbmo\Data\Pareto_ranges.xlsx', sheet_name='Sheet1') 


    #Reading the ideal and nadir points from the data frame
    Rev_range = df.iloc[1:25,10]
    HA_range = df.iloc[1:25,11]
    CO2_range = df.iloc[1:25,12]
    DW_range = df.iloc[1:25,13]

    #Separating ideals and nadirs
    Rev_ideal = []
    Rev_nadir = []
    HA_ideal = []
    HA_nadir = []
    CO2_ideal = []
    CO2_nadir = []
    DW_ideal = []
    DW_nadir = []

    for i in range(np.size(Rev_range)):
        if Rev_range.index[i]%2==0:
            Rev_ideal.append(Rev_range[i+1])
            HA_ideal.append(HA_range[i+1])
            CO2_ideal.append(CO2_range[i+1])
            DW_ideal.append(DW_range[i+1])
        else:
            Rev_nadir.append(Rev_range[i+1])
            HA_nadir.append(HA_range[i+1])
            CO2_nadir.append(CO2_range[i+1])
            DW_nadir.append(DW_range[i+1])

    # Reading scenario names and preferences determined by the DM
    scenarios = df.ix[1:12,14]
    Rev_pref = df.iloc[1:13,15]
    HA_pref = df.iloc[1:13,16]
    CO2_pref = df.iloc[1:13,17]
    DW_pref = df.iloc[1:13,18]

    #Preferences, ideals, and nadirs matrix (scen, obj) (reading from the dataset)
    pref = np.zeros((s,k))
    prefi = np.zeros((s,k))
    ideal = np.zeros((s,k))
    nadir = np.zeros((s,k))
    for t in range(s):
        pref[t] = np.array([Rev_pref.iloc[t], HA_pref.iloc[t], CO2_pref.iloc[t], DW_pref.iloc[t]]) 
        ideal[t] = np.array([Rev_ideal[t], HA_ideal[t], CO2_ideal[t], DW_ideal[t]]) 
        nadir[t] = np.array([Rev_nadir[t], HA_nadir[t], CO2_nadir[t], DW_nadir[t]]) 
        
    for t in range(s):
        prefi[t] = np.array([Rev_pref.iloc[t], HA_pref.iloc[t], CO2_pref.iloc[t], DW_pref.iloc[t]])
        
    return pref, ideal, nadir, prefi








def gammaa(pref, ideal, nadir,s,k):
    """
    Accept pref matrix (partly filled out by the DM), ideal and nadir points, 
    then calculate gamma_i^t (for all i and t=1,...,q) as formulated in equation (2)
        

    Parameters
    ----------
    pref : numpy.ndarray
        pref matrix (partly filled out by the DM).
    ideal : numpy.ndarray
        A matrix of ideal values for all objectives in all scenarios.
    nadir : numpy.ndarray
        A matrix of estimated nadir values for all objectives in all scenarios.
    s : int
        Number of scenarios.
    k : int
        Number of objectives.

    Returns
    -------
    gamma : numpy.ndarray
        
    \gamma_i^t = \dfrac{|g_i^t - z_i^{t[ideal]}|}{|z_i^{t[nad]} - z_i^{t[ideal]}|}     

    """

    gamma = np.zeros((s,k))

    for i in range(k):
        for t in range(s):
            if math.isnan(pref[t,i]):
                pass 
            else:
                gamma[t,i] = abs(pref[t,i] - ideal[t,i])/abs(nadir[t,i] - ideal[t,i])
            
    return gamma




def goo(pref,ideal, nadir, s, k):
    """
     Calculating preference candidates for remaining s-q scenarios                 

    Parameters
    ----------
    pref : numpy.ndarray
        pref matrix (partly filled out by the DM).
    ideal : numpy.ndarray
        A matrix of ideal values for all objectives in all scenarios.
    nadir : numpy.ndarray
        A matrix of estimated nadir values for all objectives in all scenarios.
    s : int
        Number of scenarios.
    k : int
        Number of objectives.

    Returns
    -------
    go : dict
        g_i^{o(r)} = gamma_i^r (z_i^{o[nad] - z_i^{o[ideal]}) + z_i^{o[ideal] for i=1,..,k; r=1,..,q; and o in S\Q.
        
    Q : list
        List of scenarios for which DM's preferences are available. 
        e.g., Q = [0, 3, 8, 11]
        
    sq: list
        List of scenarios for which DM's preferences are unknown. 
        e.g., sq = [1, 2, 4, 5, 6, 7, 9, 10]
        

    """

    sq = [] #np.zeros((s-q)) # S\Q indices
    Q = []

    for t in range(s):
        if math.isnan(pref[t,0]):
            sq.append(t)
        else:
            Q.append(t)
 
    # go[r,i]; i=1,..,k; r=1,..,q, o in S\Q

    go = {}

    gamma = gammaa(pref, ideal, nadir, s, k)

    for o in sq:
        go["g{0}".format(o)] = np.zeros((np.size(Q),k))
        #exec(f'g{o+1} = np.zeros((q,k))') # g1, g2, g4, g5, g6, g7, g9, g10
        t = 0
        for r in Q:
            for i in range(k):
                go["g{0}".format(o)][t,i] = gamma[r,i] * (nadir[o,i] - ideal[o,i]) + ideal[o,i]
                #exec(f'g{o+1}[t,i] = gamma[r,i] * (nadir[o,i] - ideal[o,i]) + ideal[o,i]')
            t += 1   
                
    return go, Q, sq





"""
Solving an optimization problem to find the prefered candidate for preferences
    
Estimate candidates for the DM’s preferences

1. Moderate/Realistic choice

"""

# Definning objective function and constraints 
        
# Objective function
def objective(X, k, q):
    """
    Objective function for an equivalent linear optimization problem to model (3) in the paper

    Parameters
    ----------
    X : numpy.ndarray
        Decision variables
    k : int
        Number of objective functions in the original problem.
    q : int
        Number of scenarios for which DM's preferences are available. 

    Returns
    -------
    TYPE
        Objective value.

    """
    
    #Decision variables
    #for i in range(1,len(X)+1): # len(X)+1 = k+q+k*q+1
    #    exec(f'X{i} = X[i-1]')
    
    return sum(X[k+q:k+q+2*k*q])




# Constraints
"""Generilizing
for i in range(k):
    for r in range(q):
        def exec(f'constraint{ir}')(X): # i=1, r=1
            # g_i^o - (phi_{ir}^+) + (phi_{ir}^-) - gamma_i^{o(r)}
            return X[i]+X[k+q+i*r]-gamma[r,i] # in gamma r=0,3,8,11
"""

# Linearization constraints
def constraint11(X, k, q, o, go): # i=1(0), r=1(0)
    # g_i^o - phi_{ir}^+ + phi_{ir}^- - g_i^{o(r)}
    return X[0]-X[k+q]+X[k+q+1]-go['g%s' % o][0,0]

def constraint21(X, k, q, o, go): # i=2(1), r=1(0)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[1]-X[k+q+2]+X[k+q+3]-go['g%s' % o][0,1]  

def constraint31(X, k, q, o, go): # i=3(2), r=1(0)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[2]-X[k+q+4]+X[k+q+5]-go['g%s' % o][0,2]  

def constraint41(X, k, q, o, go): # i=4(3), r=1(0)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[3]-X[k+q+6]+X[k+q+7]-go['g%s' % o][0,3]


def constraint12(X, k, q, o, go): # i=1(0), r=2(1)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[0]-X[k+q+8]+X[k+q+9]-go['g%s' % o][1,0]        
        
def constraint22(X, k, q, o, go): # i=2(1), r=2(1)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[1]-X[k+q+10]+X[k+q+11]-go['g%s' % o][1,1]

def constraint32(X, k, q, o, go): # i=3(2), r=2(1)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[2]-X[k+q+12]+X[k+q+13]-go['g%s' % o][1,2]

def constraint42(X, k, q, o, go): # i=4(3), r=2(1)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[3]-X[k+q+14]+X[k+q+15]-go['g%s' % o][1,3]


def constraint13(X, k, q, o, go): # i=1(0), r=3(2)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[0]-X[k+q+16]+X[k+q+17]-go['g%s' % o][2,0]        

def constraint23(X, k, q, o, go): # i=2(1), r=3(2)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[1]-X[k+q+18]+X[k+q+19]-go['g%s' % o][2,1]

def constraint33(X, k, q, o, go): # i=3(2), r=3(2)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[2]-X[k+q+20]+X[k+q+21]-go['g%s' % o][2,2]

def constraint43(X, k, q, o, go): # i=4(3), r=3(2)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[3]-X[k+q+22]+X[k+q+23]-go['g%s' % o][2,3]


def constraint14(X, k, q, o, go): # i=1(0), r=4(3)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[0]-X[k+q+24]+X[k+q+25]-go['g%s' % o][3,0]

def constraint24(X, k, q, o, go): # i=2(1), r=4(3)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[1]-X[k+q+26]+X[k+q+27]-go['g%s' % o][3,1]

def constraint34(X, k, q, o, go): # i=3(2), r=4(3)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[2]-X[k+q+28]+X[k+q+29]-go['g%s' % o][3,2]

def constraint44(X, k, q, o, go): # i=4(3), r=4(3)
    # g_i^o + phi_{ir} - gamma_i^{o(r)}
    return X[3]-X[k+q+30]+X[k+q+31]-go['g%s' % o][3,3]
   

#-------------------------------------------
con11 ={'type':'eq', 'fun':constraint11}
con21 ={'type':'eq', 'fun':constraint21}
con31 ={'type':'eq', 'fun':constraint31}
con41 ={'type':'eq', 'fun':constraint41}
con12 ={'type':'eq', 'fun':constraint12}
con22 ={'type':'eq', 'fun':constraint22}
con32 ={'type':'eq', 'fun':constraint32}
con42 ={'type':'eq', 'fun':constraint42}
con13 ={'type':'eq', 'fun':constraint13}
con23 ={'type':'eq', 'fun':constraint23}
con33 ={'type':'eq', 'fun':constraint33}
con43 ={'type':'eq', 'fun':constraint43}
con14 ={'type':'eq', 'fun':constraint14}
con24 ={'type':'eq', 'fun':constraint24}
con34 ={'type':'eq', 'fun':constraint34}
con44 ={'type':'eq', 'fun':constraint44}




# Feasible space constraints
def constraint1(X, o, go):
    # g_i^2 - sum_{r=1}^4(lambda_r * gamma_i^{o(r)}) phi_{ir} - gamma_i^{o(r)} >= 0
    return X[0] - (X[4]*go['g%s' % o][0,0] + X[5]*go['g%s' % o][1,0] + X[6]*go['g%s' % o][2,0] + X[7]*go['g%s' % o][3,0])

def constraint2(X, o, go):
    # g_i^2 - sum_{r=1}^4(lambda_r * gamma_i^{o(r)}) phi_{ir} - gamma_i^{o(r)} >= 0
    return X[1] - (X[4]*go['g%s' % o][0,1] + X[5]*go['g%s' % o][1,1] + X[6]*go['g%s' % o][2,1] + X[7]*go['g%s' % o][3,1])

def constraint3(X, o, go):
    # g_i^2 - sum_{r=1}^4(lambda_r * gamma_i^{o(r)}) phi_{ir} - gamma_i^{o(r)} >= 0
    return X[2] - (X[4]*go['g%s' % o][0,2] + X[5]*go['g%s' % o][1,2] + X[6]*go['g%s' % o][2,2] + X[7]*go['g%s' % o][3,2])

def constraint4(X, o, go):
    # g_i^2 - sum_{r=1}^4(lambda_r * gamma_i^{o(r)}) phi_{ir} - gamma_i^{o(r)} >= 0
    return X[3] - (X[4]*go['g%s' % o][0,3] + X[5]*go['g%s' % o][1,3] + X[6]*go['g%s' % o][2,3] + X[7]*go['g%s' % o][3,3])

# Convex constraints
def constraint5(X, k, q, o, go):
    # sum_{r=1}^4(lambda_r) - 1 = 0
    return sum(X[k:k+q]) - 1 


con1 ={'type':'ineq', 'fun':constraint1}
con2 ={'type':'ineq', 'fun':constraint2}
con3 ={'type':'ineq', 'fun':constraint3}
con4 ={'type':'ineq', 'fun':constraint4}

con5 ={'type':'eq', 'fun':constraint5}


Cons = [con1, con2, con3, con4, con5, 
        con11, con21, con31, con41,
        con12, con22, con32, con42,
        con13, con23, con33, con43,
        con14, con24, con34, con44,
        ]




"""

2. Idealistic choice

"""
# Solving an optimization problem to find the prefered candidate for preferences

# Objective function
def objectivei(X, k, q):
    #Decision variables
    #for i in range(1,len(X)+1): # len(X)+1 = k+q+k*q+1
    #    exec(f'X{i} = X[i-1]')
    
    return sum(X[k+q:k+q+2*k])

# Constraints

# Linearization constraints
def constraint6(X, k, q, o, ideal): # i=1(0), 
    # g_i^o - phi_{ir}^+ + phi_{ir}^- - z_i^{o[ideal]} = 0
    return X[0]-X[k+q]+X[k+q+1]-ideal[o,0]

def constraint7(X, k, q, o, ideal): # i=2(1), 
    # g_i^o - phi_{ir}^+ + phi_{ir}^- - z_i^{o[ideal]}
    return X[1]-X[k+q+2]+X[k+q+3]-ideal[o,1]  

def constraint8(X, k, q, o, ideal): # i=3(2), 
    # g_i^o - phi_{ir}^+ + phi_{ir}^- - z_i^{o[ideal]}
    return X[2]-X[k+q+4]+X[k+q+5]-ideal[o,2]  

def constraint9(X, k, q, o, ideal): # i=4(3), 
    # g_i^o - phi_{ir}^+ + phi_{ir}^- - z_i^{o[ideal]}
    return X[3]-X[k+q+6]+X[k+q+7]-ideal[o,3]

con6 ={'type':'eq', 'fun':constraint6}
con7 ={'type':'eq', 'fun':constraint7}
con8 ={'type':'eq', 'fun':constraint8}
con9 ={'type':'eq', 'fun':constraint9}


Consi = [con1, con2, con3, con4, con5, 
        con6, con7, con8, con9,
        ]



























