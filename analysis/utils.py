import pandas as pd
import numpy as np

plotdir = "../plots"

def saveplot(fig, fileprefix):
    for ext in ['pdf', 'png']:
        fl = f"{plotdir}/{fileprefix}.{ext}"
        fig.savefig(fl, bbox_inches = 'tight')
    return


def stratify_dfcol(df, colname, value):
    #return pd_utils.select_dfrows(df, [f"$({colname}) == {value}"])
    return df.loc[df[colname] == value]


def stratify_dfcols(df, condition_list):
    for (colname, value) in condition_list:
        df = stratify_dfcol(df, colname, value)
    return df


def stratify_dfcols_in_list(df, colname, values):
    return df.loc[df[colname].isin(values)]


def remove_axis_names(df):
    return df.rename_axis(None, axis = 0).rename_axis(None, axis = 1)

def pivot_simulation_stat(ext_df, colname,
                            primary_keys = ['simulate', 'simulate.dims', 'simulate.sfix', 'simulate.pve'],
                            secondary_keys = ['DSC'],
                            unique_keys = ['simulate.sfix', 'simulate.pve'],
                            id_col = 'sim_id'):
    # Make a local copy, do not change external DataFrame
    df = ext_df.copy(deep = True)

    # Using the primary and secondary keys, create combination of unique simulation IDs.
    indices = df.set_index(primary_keys + secondary_keys).index.factorize()[0] + 1
    df.insert(loc = 0, column = id_col, value = indices)

    # Create a simulation DataFrame with unique properties of the simulations
    sim_df = df[[id_col] + unique_keys].drop_duplicates().set_index(id_col)
    sim_df = remove_axis_names(sim_df)
    sim_df[id_col] = sim_df.index

    # Pivot the DataFrame with statistics of all methods
    stat_df = df[[id_col, 'fit', colname]].pivot(index = id_col, columns = 'fit', values = colname)
    stat_df = remove_axis_names(stat_df)

    # Test the indices are correct
    pd.testing.assert_index_equal(sim_df.index, stat_df.index)

    # Return merged DataFrame
    return pd.merge(sim_df, stat_df, left_index=True, right_index=True)
