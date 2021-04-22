from sbmo import knaps_gurobi as my_knaps
# replace my.mpsolvers.knaps_gurobi -> knaps_gurobi
# if you put knaps_gurobi module in the same folder
import pandas as pd
import numpy as np

### Initialization and constants
# Folder with csv files represrnting initial problem coefficients
data_dir=r'C:\Users\babshava\SBMOP_forest\sbmo\Data'
cr_names, cr_abbr, cr_files  = list(map(list,zip(*[
["Revenues", "Rev", "Timber_revenues"],
["Habitat availability", "HA", "Combined_HA"],
["Carbon storage", "Carb", "Carbon_storage"],
["Deadwood volume", "DW", "Deadwood_volume"]
    ])))
cr_n0 = len(cr_abbr) # nr. of initial ctiteria

## Given nr. of scenario and abbreviation or nr. of initial criterion,
# returns the nr. of criterion from flattened [scenarios] x [criteria]
def f_crit_id(sc,cr):
    if type(cr) == str:
        cr = cr_abbr.index(cr)
    return sc*cr_n0 + cr

## Structure of the Dataframe for storing objective function coefficients
dd = pd.DataFrame(
    columns = pd.MultiIndex(
        levels=[[],[]], codes=[[],[]], names = ["Criterion","Regimes"]
        )
    )


### Reading data
for fi, ai in zip(cr_files,cr_abbr):
    di = pd.read_csv(f'{data_dir}\\{fi}.csv',na_values="NA")
    dd[pd.MultiIndex.from_product([[ai],di.columns])] = di
## checking integrity
d0 = None
for i, ci in enumerate(cr_abbr):
    d1 = dd[ci].applymap(pd.isna)
    if i>0 and ((d1!=d0).any(None) or any(d0.columns != d1.columns)):
        print(f"!!! {ci} differs from previous !!!")
    d0 = d1.copy()


### Initializing the multiobj. math. progr. model
M = my_knaps.knapsack_gurobi(
    var_shape = dd["Rev"].count(axis=1).values,
    obj2out = lambda dic: {ki:-dic[ki] for ki in dic}
    )

### Scenario 0: original
for ci in cr_abbr:
    M.add_obj(
        [-di.dropna().values
            for _, di in dd.loc[:,[ci]].iterrows()]
        )
### Calculate and pront Pareto ranges
M.eff_range()
d_rng = pd.DataFrame(
    columns = cr_names,index=["Nadir","Ideal"],
    data = np.array(
        [my_knaps.obj_dict2vect(vi) for vi in [M.out_nadir,M.out_ideal]]
        )
    )
print(d_rng)