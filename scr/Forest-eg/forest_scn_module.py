import knaps_gurobi as my_knaps
# import os
# os.environ["DEBUSSY"] = "1"
import pandas as pd
from copy import copy
import random
from math import sqrt
from itertools import product
import json
from os import path as os_path

### Initialization and constants
total_area = 68700 # ha, from the paper
init_sc_name = "Basic scenario"
random.seed(20201029) # For repeatability of experiments

## Folder with csv files represrnting initial problem coefficients
# Localizing folders depending on the user
data_dir_b=r'C:\Users\babshava\Dropbox\Current works\DM with incomplet preferences\Forestry example\OneDrive_2020-12-18\Forest problem\Data'
calc_dir_b=r'C:\Users\babshava\Dropbox\Current works\DM with incomplet preferences\Forestry example\OneDrive_2020-12-18\Forest problem\Calc'
data_dir_d='/home/dp/Dropbox/Scenarios_MOO/Forest problem/Data'
calc_dir_d='/home/dp/Dropbox/Scenarios_MOO/Forest problem/Calc'
if os_path.isdir(data_dir_b):
    data_dir = data_dir_b
    calc_dir = calc_dir_b
else:
    data_dir = data_dir_d
    calc_dir = calc_dir_d

## Names of criteria:
# names to display, abbreviations to index data tables, data file names
cr_names, cr_abbr, cr_files = list(map(list,zip(*[
["Revenues", "Rev", "Timber_revenues"],
["Habitat availability", "HA", "Combined_HA"],
["Carbon storage", "Carb", "Carbon_storage"],
["Deadwood volume", "DW", "Deadwood_volume"]
    ])))
cr_n0 = len(cr_abbr) # nr. of initial ctiteria

## Scenarios = (ARSs of type 0) x (ARSs of type 1) x ...
#  where ARS is Aspect-Related Scenario
sc_types = [
    ["","Emissions B1", "Emissions A2"],
    ["", "Thinning subsidy"],
    ["", "Compens. for protection"]
    ]
types_lens = [len(li) for li in sc_types]
types_n = len(sc_types)
sc_nrs_list = list(product(*[range(i) for i in types_lens]))
sc_n = len(sc_nrs_list)

### Functions

## Generic function to modify coeffs based on ranges of random multipliers
#   cdf = dataframe of coeffitients matrix, 
#   rdic = {regime name: range as (rmin,rmax)}, or (rmin,rmax) for all regimes
def modif_range(cdf,rdic):
    # from dict or single range to list of ranges for speeding up
    ranges_list = \
        [rdic[ki] for ki in cdf.columns] if isinstance(rdic, dict) else \
        [rdic for _ in cdf.columns]
    # Init the modified coefficients dataframe
    mod_df = pd.DataFrame(columns = cdf.columns, index = cdf.index)
    # Modify iterativevly by rows
    for i,row_i in cdf.iterrows():
        rnd = random.random()
        for j,((minr,maxr),vi) in enumerate(zip(ranges_list,row_i.array)):
            mod_df.iat[i,j] = vi * (
                minr + (maxr-minr)*sqrt(rnd*random.random())
                    )
    return mod_df

## Miscellaneous

# Given list [for each type, ARS's number], Returns the combined scenario name
def sc_types2name(v):
    str_sep = " + "
    if v == tuple(0 for _ in range(types_n)):
        return init_sc_name
    return \
        str_sep.join(li[ti] for ti, li in zip(v,sc_types) if li[ti]!="")

# Given list of scenario types (name or nr), 
# Returns the nr. of scenario in flattened scenarios list
def sc_types2nr(v):
    n = 0 # final number
    mult = 1 # cumulative multiplier
    for ti, ni in reversed( list(zip(v,types_lens)) ):
         n += ti * mult
         mult *= ni
    return n
# Given scenario ordinary nr, return list of ARS nrs for all types
def sc_nr2types(n):
    return sc_nrs_list[n]

# Given scenario (nr. or ARSs list) and init. criterion (nr. or abbreviation),
# Returns the nr. of criterion in the flattened [scenarios] x [criteria]
def sc_cr_nr(sc, cr):
    if hasattr(sc,"__len__"): # ARSs list -> nr. of scenario
        sc = sc_types2nr(sc)
    if type(cr) == str:
        cr = cr_abbr.index(cr)
    return sc*cr_n0 + cr

#-------------- Problem data

res_table = pd.DataFrame(
    index=pd.MultiIndex.from_tuples([],names=("Scenario","Obj. vector")),
    columns = cr_names
    )
for i in range(sc_n):
    vi = sc_nr2types(i)
    with open(f"{calc_dir}\\eff_range_{i}.txt") as f:
        d = json.load(f)
    res_table.loc[(sc_types2name(vi),"Nadir"),:] = pd.Series(d["out_nadir"])
    res_table.loc[(sc_types2name(vi),"Ideal"),:] = pd.Series(d["out_ideal"])

### Initialize the structure of obj. f. coefficients for all scenarios
#  Dictionary with keys as tuples(for each scenario type, # of ARS),
#  each value is a dictionary {abbr. of objective: matrix as a dataframe}
C_all = {}
# Coefficients for the initial scenario, as part of the structure
C0 = C_all[tuple(None for _ in types_lens)] = {}


### Preparing initial data 
## Reading initial coefficient matrices
for fi, ai in zip(cr_files,cr_abbr):
    di = pd.read_csv(f'{data_dir}\\{fi}.csv',na_values="NA")
    C0[ai] = di
    if fi == cr_files[0]: # setting technical dimensions
        stands_n, regimes_n = di.shape
        reg_active_n = di.count(axis=1).values
        reg_names = list(di.columns)
        
## Checking for integrity - NaN values at same positions of the matrices
d0 = None # initialize the Boolean mask of NaN values
for i, ci in enumerate(cr_abbr):
    d1 = C0[ci].applymap(pd.isna)
    if i>0 and ((d1!=d0).any(None) or any(d0.columns != d1.columns)):
        print(f"!!! {ci} differs from previous !!!")
    d0 = d1.copy()
del(d0)
del(d1)

## Inferring stand areas
# For a df, returns df of shares of values in column sums, negative ignored
def df_shares(df): 
    d = df.copy()
    d[d<0] = float("nan")
    return d/d.sum()
# Shares of coefficients in columns for all coeffs. matrices
d_temp = pd.concat([df_shares(di) for _,di in C0.items()],axis=1)
areas = d_temp.mean(axis=1)*total_area
del(d_temp)
                  

### Generating problem data for different scenarios
# For each scenario type, a function is defined
# (dict. of problem coefficients, ARS number) -> dict. of modified coefficients

## Thinning subsidies - changes only revenues coeffs
def modif_thin_subs(C, ac_nr):
    if ac_nr == 0:
        return C
    subs_ha = 430
    # For objectives which are not modified, coeff. matrices are linked
    C_mod = {ki: C[ki] for ki in cr_abbr[1:]}
    # The modified matrix is copied
    C_mod["Rev"] = copy(C["Rev"])
    # Modification of revenues
    for ai in ["BAU", "EXT10", "EXT30", "GTR30"]:
        C_mod["Rev"][ai] += areas*subs_ha
    return C_mod

## Compensation for conservation - changes only revenues coeffs
def modif_conserv_compens(C, ac_nr):
    if ac_nr == 0:
        return C
    # Compensation for postponing harvest for one year
    subs_ha_year = 30 
    # For 3 regimes, for how many years is harvest postponed 
    regimes_years = [["EXT10",10], ["EXT30",30], ["SA",50]]
    # For objectives which are not modified, coeff. matrices are linked
    C_mod = {ki: C[ki] for ki in cr_abbr[1:]}
    # The modified matrix is copied
    C_mod["Rev"] = copy(C["Rev"])
    # Modification of revenues, based on years of no harvest
    for ai, yi in regimes_years:
        C_mod["Rev"][ai] += areas*subs_ha_year*yi
    return C_mod

## Climate scenarios - changes all coeffs
def modif_climate(C, ac_nr):
    if ac_nr == 0:
        return C
    # Inits and constants
    # For Revenues and Carbon objectives, under HadCM2, 2 groups of regimes
    range_rev_BT = (1.0500, 1.1053)
    range_rev_UT = (1.1304, 1.1408)
    range_carb_BT = (1.0500, 1.1053)
    range_carb_UT = (1.1304, 1.1408)
    HadCM2B1, HadCM2A2 = (0.75,1.5) # multipliers from HadCM2 to our regimes
    regimes_BT = ['BAU', 'EXT10', 'EXT30', 'GTR30'] 
    regimes_UT = ['SA', 'NTSR', 'NTLR']
    # For HA objective - same for all regimes
    range_ha_B1 = (0.9,1.1)
    range_ha_A2 = (0.8,1.2)
    # For Deadwood objective - under A2, different for SA regime
    range_dw_SA = (1.33,1.88)
    range_dw_others = (1.19,1.51)
    dw_B2_mult = 0.5 # Reduction from A2 to B1 scenario
    # Settings depending on climate scenario
    if ac_nr == 1: # B1 scenario
        HadCM2mult = HadCM2B1
        range_HA = range_ha_B1
        range_dw_SA, range_dw_others = [
            [(v-1)*dw_B2_mult + 1 for v in t]
            for t in [range_dw_SA, range_dw_others]
                ]

    else:          # A2 scenario
        HadCM2mult = HadCM2A2
        range_HA = range_ha_A2
    range_rev_BT, range_rev_UT, range_carb_BT, range_carb_UT = [
        [(v-1)*HadCM2mult + 1 for v in t]
        for t in 
            [range_rev_BT, range_rev_UT, range_carb_BT, range_carb_UT]
            ]

    # Input information per objective
    ranges_input = {}
    ranges_input["Rev"] = {ki: range_rev_BT for ki in regimes_BT}
    ranges_input["Rev"].update({ki: range_rev_UT for ki in regimes_UT})
    ranges_input["Carb"] = {ki: range_carb_BT for ki in regimes_BT}
    ranges_input["Carb"].update({ki: range_carb_UT for ki in regimes_UT})
    ranges_input["HA"] = range_HA
    ranges_input["DW"] = {ki:range_dw_others for ki in reg_names}
    ranges_input["DW"]["SA"] = range_dw_SA
    # Create modified coefficients for individual objectives    
    return {
        ki: modif_range(C[ki],ranges_input[ki])
            for ki in cr_abbr
            }


### Generating coefficients for all scenarios
modif_functions = [modif_climate, modif_thin_subs, modif_conserv_compens]
for ti, (ni,fi) in enumerate(zip(types_lens,modif_functions)):
    # ti - type nr, ni - nr of ARSs of this type, fi - modifier function
    for ki in list(C_all):
        # ki - key of existing coeffitients set, that will be modified
            for sci in range(ni):
                # sci - ARS number
                knew = tuple( # modified key where ti-th element = sci
                    sci if i == ti else j
                        for i,j in enumerate(ki)
                        )
                C_all[knew] = fi(C_all[ki],sci)
            del(C_all[ki])

### Returns the knapsack problem object, with Pareto range read from a file
##  scenario = ...
#       None - combined objectives for all scenarios, named "sc#,cr_abbr"
#       int - for the scenario of given nr.
#       tuple of int - scenario defined by ARSs nrs.
def gur_probl(scenario = None):
    M = my_knaps.knapsack_gurobi(
        var_shape = reg_active_n,
        obj2out = lambda dic: {ki:-dic[ki] for ki in dic}
        )
    # Combined problem
    if scenario is None:
        for i in range(sc_n):
            vi = sc_nr2types(i)
            for ci in cr_abbr:
                M.add_obj(
                    [ -row_i.dropna().values for _, row_i in 
                            C_all[vi][ci].iterrows() ],
                    name=f"{i},{ci}"
                    )
        with open(f"{calc_dir}\\eff_range_all.txt") as f:
            M.set_range(json.load(f))
        return M
    # Scenario-related problem
    if hasattr(scenario,"__len__"):
        n = sc_types2nr(scenario)
        v = scenario
    else:
        n = scenario
        v = sc_nr2types(scenario)
    for ci in cr_abbr:
        M.add_obj(
            [ -row_i.dropna().values for _, row_i in 
                    C_all[v][ci].iterrows() ],
            name=ci
            )
    with open(f"{calc_dir}\\eff_range_{n}.txt") as f:
        M.set_range(json.load(f))
    return M




## Convert obj.vector as DataFrame to dict. format (for knapsack backage)
#  If scenario nr. is given, then obj. vector is for given scenario
#  note multiplication by -1
def df2objvector(df, scenario = None):
    if scenario is None:
        return {
            f"{sci},{cri}": -vi
                for sci, rowi in df.iterrows()
                    for cri, vi in rowi.iteritems()
            }
    return { cri: -vi for cri, vi in df.loc[scenario].iteritems() }  

## Convert obj.vector from dict. to DataFrame format (if it is for all scenarios)
#  or to Series (if it is for one scenario)
def objvector2df(dic):
    # The case where obj. vector is given for all scenarios
    if "," in list(dic.keys())[0]:
        d_out = pd.DataFrame(index=range(sc_n),columns=cr_abbr)
        for ki,vi in dic.items():
            xy = ki.split(",")
            d_out.loc[int(xy[0]),xy[1]] = -vi
        d_out["Scenario"] = d_out.index
    # The case where obj. vector is given for one scenario
    else:
        d_out = pd.Series(index=cr_abbr,dtype='float64')
        for ki,vi in dic.items():
            d_out[ki] = -vi
    return d_out

## Read ref. point from a csv file and represent it as DataFrame, where
#       columns = criteria abbreviations, index = scenario nrs. (from 0)
#  Purpose: for displaying/examining/analysing/manipulating refpoints
def read_refpoint_df(fname):
    df = pd.read_csv(fname).set_index("Scenario")
    # Basic consistency check: return error if the names of criteria are wrong
    assert set(df.columns) == set(cr_abbr)
    return df

## Reads csv file, returns a ref. point as dictionary, for knapsack module
def read_refpoint(fname, scenario=None ):
    return df2objvector(
        read_refpoint_df(fname), scenario=scenario 
        )
 

##### Code to produce Pareto ranges for separate problems and the combined one
# ### Pareto ranges separately by scenarios
# res_table = pd.DataFrame(
#     index=pd.MultiIndex.from_tuples([],names=("Scenario","Obj. vector")),
#     columns = fsc.cr_names
#     )
# for i in range(fsc.sc_n):
#     vi = fsc.sc_nr2types(i)
#     print("\n**********\nCalculating for",vi)
#     # Init 4-objective problem
#     M = my_knaps.knapsack_gurobi(
#         var_shape = fsc.reg_active_n,
#         obj2out = lambda dic: {ki:-dic[ki] for ki in dic}
#         )
#     # Add objectives
#     for ci,ni in zip(fsc.cr_abbr,fsc.cr_names):
#         M.add_obj(
#             [ -row_i.dropna().values for _, row_i in 
#                     fsc.C_all[vi][ci].iterrows() ],
#             name=ni
#             )
#     # Calculate and save Pareto ranges
#     M.eff_range()
#     res_table.loc[(fsc.sc_types2name(vi),"Nadir"),:] = pd.Series(M.out_nadir)
#     res_table.loc[(fsc.sc_types2name(vi),"Ideal"),:] = pd.Series(M.out_ideal)
#     with open(f"{fsc.calc_dir}\\eff_range_{i}.txt","w") as f:
#         json.dump(M.get_range(),f)

# ### Pareto ranges for the combined problem
# M = my_knaps.knapsack_gurobi(
#     var_shape = fsc.reg_active_n,
#     obj2out = lambda dic: {ki:-dic[ki] for ki in dic}
#     )
# # Adding objectives
# for i in range(fsc.sc_n):
#     vi = fsc.sc_nr2types(i)
#     for ci,ni in zip(fsc.cr_abbr,fsc.cr_names):
#         M.add_obj(
#             [ -row_i.dropna().values for _, row_i in 
#                     fsc.C_all[vi][ci].iterrows() ],
#             name=f"{i},{ni}"
#             )
# # Calc. efficient range
# M.eff_range()
# # Writing results
# res_table = pd.DataFrame(
#     index=pd.MultiIndex.from_tuples([],names=("Scenario","Obj. vector")),
#     columns = fsc.cr_names
#     )
# with open(f"{fsc.calc_dir}\\eff_range_all.txt","w") as f:
#     json.dump(M.get_range(),f)
# for i in range(fsc.sc_n):
#     vi = fsc.sc_nr2types(i)
#     for ni in fsc.cr_names:
#         obj_name = f"{i},{ni}"
#         res_table.loc[(fsc.sc_types2name(vi),"Nadir"),ni] = \
#             M.out_nadir[obj_name]
#         res_table.loc[(fsc.sc_types2name(vi),"Ideal"),ni] = \
#             M.out_ideal[obj_name]
