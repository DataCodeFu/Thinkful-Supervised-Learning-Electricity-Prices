"""

"""




# Import Python libraries.
import csv
import os
import calendar
import itertools as it
import warnings
import io
import math
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from scipy.stats import (
    ttest_ind,
    pearsonr,
    zscore,
    norm,
    jarque_bera,
    normaltest,
    shapiro,
    boxcox,
    bartlett,
    levene
)
from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR 
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import (
      mean_absolute_error
)
from sklearn.linear_model import (
    LogisticRegression, 
    LinearRegression, 
    Ridge, 
    RidgeCV, 
    Lasso, 
    LassoCV, 
    ElasticNet, 
    ElasticNetCV 
)
from statsmodels.tools.eval_measures import mse, rmse
import statsmodels.formula.api as smf
from IPython.display import Image
import dataframe_image as dfi


# ----- Set global options in libraries and Python environment. -----
# %matplotlib inline
warnings.filterwarnings('ignore')
pd.set_option('use_inf_as_na', True)
pd.options.display.float_format = '{:,.3f}'.format
pd.set_option('display.max_rows', 1000)  # Int or None
pd.set_option('display.min_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
sns.set(style="ticks", color_codes=True)




"""
    Generates and outputs an image of a DataFrame to the display.  Great for PDF'd Jupyter Notebooks.
"""
def df_img_out(df, img_name):
    dfi.export(df, img_name + '.png')
    display(Image(img_name + '.png'))




"""
    Convert to long-form data format, split description column, and
    remove either regional- or state-based data points (no overlap).
        folder_df_nz_melted = short_to_long_form(eia_raw_data_df)
"""
def short_to_long_form_and_wrangle(df, loc_method='state'):
    non_date_columns_idx = pd.to_datetime(df.columns, format='%b %Y', errors='coerce').isnull()
    date_columns_idx = pd.to_datetime(df.columns, format='%b %Y', errors='coerce').notnull()
    non_date_columns = df.columns[non_date_columns_idx]
    date_columns = df.columns[date_columns_idx]
    df_melted = df.melt(id_vars=non_date_columns.to_list(), 
                        var_name='year_month', value_name='values')
    df_melted['year_month'] = pd.to_datetime(df_melted['year_month'])
    df_melted = df_melted.sort_values(['csv_table_names', 'year_month', 'description'])\
        .reset_index(drop=True)
    
    # Additional data cleaning items on the combined DataFrame, split location and sector cols.
    split_description = df_melted['description'].str.split(pat=': ', expand=True)
    split_description_df = pd.DataFrame(split_description)
    split_description_df.rename(columns={0: 'location', 1: 'sector'}, inplace=True)
    
    # Convert to long-form data format.
    df_melted.insert(loc=1, column='location', value=split_description_df['location'])
    df_melted.insert(loc=2, column='sector', value=split_description_df['sector'])
    df_melted.drop(columns=['description'], inplace=True)
    df_melted.index.rename('index', inplace=True)
    for col in df_melted.select_dtypes(include='object'):
        df_melted[col] = df_melted[col].str.strip()
    
    # Split csv_table_names into variable and energy_type columns.
    split_description2 = df_melted['csv_table_names'].str.split(pat='_for_', expand=True, n=1)
    split_description_df2 = pd.DataFrame(split_description2)
    split_description_df2.rename(columns={0: 'variable', 1: 'energy_type'}, inplace=True)
    df_melted.insert(loc=0, column='variable', value=split_description_df2['variable'])
    df_melted.insert(loc=1, column='energy_type', value=split_description_df2['energy_type'])
    df_melted['energy_type'] = df_melted['energy_type'].fillna('electricity')
    df_melted['units'] = df_melted['units'].str.replace(' ', '_')
    df_melted['variable'] = df_melted['variable'].str.cat(df_melted['units'], sep='_')
    df_melted = df_melted.drop(columns=['csv_table_names', 'units'])
    
    # Energy type information was in the sector column of the fossil fuel reports.
    #     - Fossil_fuel_stocks_in_electricity_generation_thousand_barrels
    #     - Fossil_fuel_stocks_in_electricity_generation_thousand_tons
    df_mask = ( df_melted['sector'].isin(['petroleum liquids', 'coal', 'petroleum coke']) )
    df_melted.loc[df_mask, 'energy_type'] = df_melted.loc[df_mask]['sector']
    df_melted.loc[df_mask, 'sector'] = 'all sectors'
    
    # Cutoff data before 2008, no useful fuel data before then.
    df_melted = df_melted[~df_melted['year_month'].isin(pd.date_range(start='20000101', end='20071231'))]
    
    # Rename 'all commercial' and 'all industrial' by removing 'all'
    df_melted['sector'] = df_melted['sector'].replace('all commercial', 'commercial')
    df_melted['sector'] = df_melted['sector'].replace('all industrial', 'industrial')
    
    # Replace electric power sector with all sectors for:
    #     Average_cost_of_fossil_fuels_in_electricity_generation_per_Btu_dollars_per_million_Btu.
    bool_idx = ( df_melted['variable'].isin(
                      ['Average_cost_of_fossil_fuels_in_electricity_generation_per_Btu_dollars_per_million_Btu']) &
                  df_melted['sector'].isin(['electric power']) )
    df_melted.loc[bool_idx, 'sector'] = 'all sectors'
    # Sort DataFrame.
    df_melted = df_melted.sort_values(
                by=['variable', 'location', 'energy_type', 'sector', 'year_month'], ascending=True)
    # Drop regions, they are duplicates of groups of states (or drop states).
    regions = [
        'United States',
        'East North Central',
        'Pacific Contiguous',
        'Mountain',
        'South Atlantic',
        'West South Central',
        'West North Central',
        'Middle Atlantic',
        'East South Central',
        'New England',
        'Pacific Noncontiguous'
    ]
    df_mask_location_regions = ( df_melted['location'].isin(regions[1:]) )
    df_mask_location_states = ( ~df_melted['location'].isin(regions) )
    eia_df_regions = df_melted[df_mask_location_regions]
    eia_df_states = df_melted[df_mask_location_states]
    return eia_df_states if loc_method == 'state' else eia_df_regions
# folder_df_nz_long = short_to_long_form_and_wrangle(eia_raw_data_df)
# display(folder_df_nz_long.head(2))




"""
    Convert to time-series with dates as x-axis.
        folder_df_nz_long_sct = correct_section_classification(folder_df_nz_long)
"""
def correct_section_classification(df_long_input):
    df_midx = df_long_input.set_index(
        ['variable', 'energy_type', 'location', 'sector', 'year_month'])
    df_midx_pvt = df_midx.drop(columns=['source key']).unstack('sector')

    # Variables which have excess aggregated sector-level data.
    var_list = [
        'Receipts_of_fossil_fuels_by_electricity_plants_thousand_barrels',
        'Receipts_of_fossil_fuels_by_electricity_plants_thousand_mcf',
        'Receipts_of_fossil_fuels_by_electricity_plants_thousand_tons'
    ]
    df_midx_pvt_sect = df_midx_pvt.copy(deep=True)

    # Fill in empty electric utility values with electric power total less any independent power producers.
    bool_idx1 = ( df_midx_pvt_sect.index.get_level_values('variable').isin(var_list) &
                  df_midx_pvt_sect[('values', 'electric power')].notna() &
                  df_midx_pvt_sect[('values', 'electric utility')].isnull() )
    zero_nans_df1 = \
        df_midx_pvt_sect.loc[bool_idx1, ('values', 'independent power producers')].copy(deep=True)
    df_midx_pvt_sect.loc[bool_idx1, ('values', 'electric utility')] = \
        df_midx_pvt_sect.loc[bool_idx1, ('values', 'electric power')].values \
        - zero_nans_df1.replace(np.nan, 0).values

    # Fill in empty independent power producers if electric utility is not NaN with electric power total.
    bool_idx2 = ( df_midx_pvt_sect.index.get_level_values('variable').isin(var_list) &
                  df_midx_pvt_sect[('values', 'electric power')].notna() &
                  df_midx_pvt_sect[('values', 'electric utility')].notna() &
                  df_midx_pvt_sect[('values', 'independent power producers')].isnull() )
    zero_nans_df2 = \
        df_midx_pvt_sect.loc[bool_idx2, ('values', 'electric utility')].copy(deep=True)
    df_midx_pvt_sect.loc[bool_idx2, ('values', 'independent power producers')] = \
        df_midx_pvt_sect.loc[bool_idx2, ('values', 'electric power')].values \
        - zero_nans_df2.replace(np.nan, 0).values

    # Drop electric power as excess aggregation of data and stack sector in front of year_month.
    df_midx_pvt_sect_drop = df_midx_pvt_sect.drop(columns=[('values', 'electric power')])
    df_midx_pvt_sect_drop_pvt = df_midx_pvt_sect_drop.unstack('year_month')\
                                .stack('sector').stack('year_month').reset_index()
    
    # Resort the DataFrame and return it with corrected sections.
    df_midx_pvt_sect_drop_pvt = df_midx_pvt_sect_drop_pvt.sort_values(
            by=['variable', 'location', 'energy_type', 'sector', 'year_month'], ascending=True)
    
    return df_midx_pvt_sect_drop_pvt
# folder_df_nz_long_sct = correct_section_classification(folder_df_nz_long)
# display(folder_df_nz_long_sct.head(2))




"""
    Correct energy labels and convert energy units all into MMBtu.
"""
# Net generation heat rates (Btu/KWh) for coal, petroleum, natural gas, nuclear, fossil fueled, electricity (renewable)
#     Net generation (MWhs) * 1000 * Heat rate (Btu/KWh)  =  Required Btu
def correct_energy_units(df):
    for col in ['variable', 'energy_type', 'sector']:
        df[col] = df[col].str.strip()
    df['energy_type'] = df['energy_type'].replace('petroleum coke', 'petroleum_coke')
    df['energy_type'] = df['energy_type'].replace('petroleum liquids', 'petroleum_liquids')
    mmbtu_per_coal_ton = 18.875
    mmbtu_per_petroleum_coke_ton = 24.8000
    mmbtu_per_mcf_natural_gas = 0.001037
    mmbtu_per_petroleum_liquid_barrel = 5.6980
    
    # Convert natural gas from mcf 000's to MMBtu.
    #     natural_gas_conversion(eia_df_conv1.loc[bool_idx_ng, 'values'])
    def natural_gas_conversion_mmbtu(df):
        return df * mmbtu_per_mcf_natural_gas
    
    # Convert coal from ton 000's to MMBtu.
    def coal_conversion_mmbtu(df):
        return df * mmbtu_per_coal_ton
    
    # Convert petroleum coke from ton 000's to MMBtu.
    def pet_coke_conversion_mmbtu(df):
        return df * mmbtu_per_petroleum_coke_ton
    
    # Convert petroleum liquid from barrel 000's to MMBtu.
    def pet_liquid_conversion_mmbtu(df):
        return df * mmbtu_per_petroleum_liquid_barrel
    variable_conversions = {
        'Consumption_by_electricity_generation_thousand_Mcf_ng': \
        {   'variable':    ['Consumption_by_electricity_generation_thousand_Mcf'],
            'energy_type': ['natural_gas'],
            'new_name':    'Consumption_by_electricity_generation_MMBtu',
            'conversion':  natural_gas_conversion_mmbtu
        },
        'Receipts_of_fossil_fuels_by_electricity_plants_thousand_mcf_ng': \
        {   'variable':    ['Receipts_of_fossil_fuels_by_electricity_plants_thousand_mcf'],
            'energy_type': ['natural_gas'],
            'new_name':    'Receipts_of_fossil_fuels_electricity_generation_MMBtu',
            'conversion':  natural_gas_conversion_mmbtu
        },
        'Consumption_by_electricity_generation_thousand_tons_coal': \
        {   'variable':    ['Consumption_by_electricity_generation_thousand_tons'],
            'energy_type': ['coal'],
            'new_name':    'Consumption_by_electricity_generation_MMBtu',
            'conversion':  coal_conversion_mmbtu
        },
        'Receipts_of_fossil_fuels_by_electricity_plants_thousand_tons_coal': \
        {   'variable':    ['Receipts_of_fossil_fuels_by_electricity_plants_thousand_tons'],
            'energy_type': ['coal'],
            'new_name':    'Receipts_of_fossil_fuels_electricity_generation_MMBtu',
            'conversion':  coal_conversion_mmbtu
        },
        'Fossil_fuel_stocks_in_electricity_generation_thousand_tons_coal': \
        {   'variable':    ['Fossil_fuel_stocks_in_electricity_generation_thousand_tons'],
            'energy_type': ['coal'],
            'new_name':    'Fossil_fuel_stocks_in_electricity_generation_MMBtu',
            'conversion':  coal_conversion_mmbtu
        },
        'Consumption_by_electricity_generation_thousand_tons_pet': \
        {   'variable':    ['Consumption_by_electricity_generation_thousand_tons'],
            'energy_type': ['petroleum_coke'],
            'new_name':    'Consumption_by_electricity_generation_MMBtu',
            'conversion':  pet_coke_conversion_mmbtu
        },
        'Receipts_of_fossil_fuels_by_electricity_plants_thousand_tons_pet': \
        {
            'variable':    ['Receipts_of_fossil_fuels_by_electricity_plants_thousand_tons'],
            'energy_type': ['petroleum_coke'],
            'new_name':    'Receipts_of_fossil_fuels_electricity_generation_MMBtu',
            'conversion':  pet_coke_conversion_mmbtu
        },
        'Fossil_fuel_stocks_in_electricity_generation_thousand_tons_pet': \
        {   'variable':    ['Fossil_fuel_stocks_in_electricity_generation_thousand_tons'],
            'energy_type': ['petroleum_coke'],
            'new_name':    'Fossil_fuel_stocks_in_electricity_generation_MMBtu',
            'conversion':  pet_coke_conversion_mmbtu
        },
        'Consumption_by_electricity_generation_thousand_barrels_pet_liq': \
        {   'variable':    ['Consumption_by_electricity_generation_thousand_barrels'],
            'energy_type': ['petroleum_liquids'],
            'new_name':    'Consumption_by_electricity_generation_MMBtu',
            'conversion':  pet_liquid_conversion_mmbtu
        },
        'Receipts_of_fossil_fuels_by_electricity_plants_thousand_barrels_pet_liq': \
        {   'variable':    ['Receipts_of_fossil_fuels_by_electricity_plants_thousand_barrels'],
            'energy_type': ['petroleum_liquids'],
            'new_name':    'Receipts_of_fossil_fuels_electricity_generation_MMBtu',
            'conversion':  pet_liquid_conversion_mmbtu
        },
        'Fossil_fuel_stocks_in_electricity_generation_thousand_barrels_pet_liq': \
        {   'variable':    ['Fossil_fuel_stocks_in_electricity_generation_thousand_barrels'],
            'energy_type': ['petroleum_liquids'],
            'new_name':    'Fossil_fuel_stocks_in_electricity_generation_MMBtu',
            'conversion':  pet_liquid_conversion_mmbtu
        }
    }
    for var, book in sorted(variable_conversions.items()):
        bool_idx = ( df['variable'].isin(book['variable']) &
                     df['energy_type'].isin(book['energy_type']) )
        df.loc[bool_idx, 'values'] = \
            df.loc[bool_idx, 'values'].apply(book['conversion'])
        df.loc[bool_idx, 'variable'] = \
            df.loc[bool_idx, 'variable'].str.replace(book['variable'][0], str(book['new_name']))
    
    # Remove all sectors as it is redundant data in features, excluding fossil_fuel_stocks and retail sales KWhs.
    keep_all_sectors_vars = [
        'Average_retail_price_of_electricity_cents_per_kilowatthour',
        'Retail_sales_of_electricity_million_kilowatthours',
        'Fossil_fuel_stocks_in_electricity_generation_MMBtu',
        'Fossil_fuel_stocks_in_electricity_generation_thousand_tons',
        'Fossil_fuel_stocks_in_electricity_generation_thousand_barrels'
    ]
    df = df[ ~ ( df['sector'].str.strip().isin(['all sectors', 'transportation'])
                 & ~ df['variable'].str.strip().isin(keep_all_sectors_vars) ) 
           ].reset_index(drop=True)
    
    # Remove transportation and detailed sectors from target variable and related feature, only keep 'all sectors'.
    df = df[ ~ ( df['sector'].str.strip().isin(['commercial', 'industrial', 'residential', 'transportation'])
                 & df['variable'].str.strip().isin(['Average_retail_price_of_electricity_cents_per_kilowatthour',
                                                    'Retail_sales_of_electricity_million_kilowatthours']) )
           ].reset_index(drop=True)
    
    # Combine non-thermal-fuel-related energy types (only have generation data).
    renewable_types = [
        'biomass',
        'conventional_hydroelectric',
        'all_utility_scale_solar',
        'geothermal',
        'hydro_electric_pumped_storage',
        'nuclear',
        'other',
        'other_gases',
        'other_renewables',
        'wind'
    ]
    df['energy_type'] = df['energy_type'].replace(renewable_types, 'renewable_and_other')
    df = df.groupby(['variable', 'energy_type', 'location', 'sector', 'year_month']).agg('sum')  
    
    # Replace zeros with np.nan.
    df = df.replace(0, np.nan)
    df = df.reset_index(drop=False)
    return df
# folder_df_nz_long_sct_conv = correct_energy_units(folder_df_nz_long_sct)




"""
    Show variables and related sector categories.
"""
def show_category_breakdown(df, primary_col, pivot_cols):
    graph_rows, graph_cols = 1, len(pivot_cols)
    fig, axs = plt.subplots(figsize=(5 * graph_cols, 4 * graph_rows), 
                            nrows=graph_rows, ncols=graph_cols,
                            sharex='none', sharey='none') # sharex='col', sharey='row'
    for col in pivot_cols:
        c = pivot_cols.index(col)
        graph_df = pd.DataFrame(df[col].value_counts())
        graph_height = round( np.log(graph_df.index.size + 1)
                              + np.sqrt(graph_df.index.size + 1), 0 )
        if graph_df.index.size <= 25:
            graph_df.plot(kind='barh', ax=axs[c]) # figsize=(7, graph_height) )
        else:
            graph_df.plot(kind='barh', ax=axs[c])
            axs[c].set_yticklabels([])
            # axs[c].set_yticks([])
            # axs[c].set_ylabel([])
        # Show unique values for features in a table.
        dff = pd.DataFrame()
        for var in df[primary_col].unique():
            bool_idx = ( df[primary_col] == var )
            unique_sectors_for_var = pd.Series(df[bool_idx][col].unique())\
                                               .sort_values().rename(var)
            dff = pd.concat([dff, unique_sectors_for_var], axis='columns')
        dff = dff.sort_values(by=dff.columns.to_list()).transpose().sort_values(by=[0, 1, 2])
        df_img_out(dff.head().iloc[:, :8], 'show_category_breakdown')
    plt.tight_layout()
    plt.show()
# show_category_breakdown(folder_df_nz_long_sct_conv, 'variable', ['location', 'energy_type', 'sector'])
# variable_avg_df.plot(title='Variables Over Time', legend=show_legend, figsize=(9, 6))




"""
    Look at variable results on grouped data sets based on category variables.
"""
def show_agg_stats_by_category(df, category_vars,
                               agg_stats=['min', 'mean', 'max', 'count', 'sum']):
    for category_var in category_vars:
        df_img_out(df.groupby(['variable', category_var]).agg(agg_stats), 'show_agg_stats_by_category')
# show_agg_stats_by_category(folder_df_nz_long_sct_conv, ['energy_type', 'sector'],
#                            ['min', 'mean', 'max', 'count', 'sum'])




"""
    Convert to time-series with dates as x-axis.
        folder_df_nz_melted = short_to_long_form(eia_raw_data_df)
"""
def long_form_to_xdate(df):
    df_dups = df.index.duplicated()
    if np.sum(df_dups) > 1:
        print("="*5, "Duplicates", "="*5)
        display(df[df.index.duplicated()])
        display(df[df.index.duplicated()].sum())
    xdate_df = df.reset_index(drop=False)\
                 .drop(columns=['index'])\
                 .sort_values(by=['variable', 'location', 'energy_type', 
                                  'sector', 'year_month'], ascending=True)\
                 .set_index(['variable', 'location', 'energy_type', 'sector', 'year_month'])
    xdate_dups = xdate_df.index.duplicated()
    if np.sum(xdate_dups) > 1:
        print("="*5, "Duplicates", "="*5)
        display(xdate_df[xdate_dups])
        display(xdate_df[xdate_dups].sum())
        for col in xdate_df[xdate_dups]:
            display(xdate_df[xdate_dups][col].value_counts())
    xdate_df = xdate_df.unstack(['variable', 'location', 'energy_type', 'sector'])
    xdate_df.columns = xdate_df.columns.droplevel(0)
    return xdate_df
# folder_df_nz_xdate = long_form_to_xdate(folder_df_nz_long_sct_conv)
# display(folder_df_nz_xdate.head(2))




"""
    Display a DataFrame's null values visually across a row x column matrix.
        null_table_graph(eia_df_encoded.replace(0, np.nan))
"""
def null_table_graph(df, fig_size=(10, 8)):
    fig0, axs0 = plt.subplots(figsize=fig_size)
    heat_map = sns.heatmap(pd.isnull(df), ax=axs0)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45, horizontalalignment='right')
    heat_map.set_xticklabels('')
    heat_map.set_yticklabels('')
    plt.show()
# null_table_graph(folder_df_nz_xdate.replace([0, np.inf, -np.inf], np.nan))
# display(folder_df_nz_xdate.head(2))




"""
    Cycle through column multi-index levels and backfill/forward-fill.
"""
def back_forward_fill(df):
    for variable in df.columns.get_level_values('variable').unique():
        for location in df.columns.get_level_values('location').unique():
            for energy_type in df.columns.get_level_values('energy_type').unique():
                df_mask = (
                        ( df.columns.get_level_values('variable') == variable ) &
                        ( df.columns.get_level_values('location') == location ) &
                        ( df.columns.get_level_values('energy_type') == energy_type )
                )
                if len(df.loc[:, df_mask].columns) > 0:
                    df = df.replace([0, np.inf, -np.inf], np.nan)\
                           .fillna(method='bfill', axis='index').fillna(method='ffill', axis='index')
                    # df.loc[:, df_mask].plot(legend=False, figsize=(9, 6))
    return df
# folder_df_nz_xdate_fill = back_forward_fill(folder_df_nz_xdate)
# print("\nShowing the difference in .describe() statistics after applying the fill assumptions.\n")
# display(folder_df_nz_xdate_fill.describe() - folder_df_nz_xdate.describe())
# null_table_graph(folder_df_nz_xdate_fill)




"""
    Cycle through column multi-index levels and drop columns with largely empty values.
"""
def drop_empty_multi_idx_cols(input_df, max_nulls=50, convert_zeros_to_nans=True):
    df = input_df.copy(deep=True)
#     if convert_zeros_to_nans:
#         df = df.replace(0, np.nan)
    for variable in df.columns.get_level_values('variable').unique():
        for location in df.columns.get_level_values('location').unique():
            for energy_type in df.columns.get_level_values('energy_type').unique():
                for sector in df.columns.get_level_values('sector').unique():
                    df_mask = (
                            ( df.columns.get_level_values('variable') == variable ) &
                            ( df.columns.get_level_values('location') == location ) &
                            ( df.columns.get_level_values('energy_type') == energy_type ) &
                            ( df.columns.get_level_values('sector') == sector )
                    )
                    if len(df.loc[:, df_mask].columns) > 0:
                        null_count = df.loc[:, df_mask].isnull().sum()
                        
                        if null_count.min() > max_nulls:
                            mi_col_name = df[variable, location, energy_type, sector].name
                            x = len(df.columns)
                            df = df.drop(columns=[mi_col_name])
                            y = len(df.columns)
                            if (x - y) != 1:
                                print("Not equal to change of 1 less row?", x, y)
    return df
# folder_df_nz_xdate_fill_drop = drop_empty_multi_idx_cols(folder_df_nz_xdate_fill)
# col_count_chg = folder_df_nz_xdate_fill_drop.columns.size - folder_df_nz_xdate_fill.columns.size
# print("Change in count of columns after dropping: {:,.0f} ({:,.0f} remaining columns)".format(
#     col_count_chg, folder_df_nz_xdate_fill_drop.columns.size))




"""
    Plot Discrete Features by Column Multi-Index.
"""
def plot_discrete_features_by_col_idx(df, target_var):
    # Show each variable average value over time.
    variable_avg_df = df.groupby(axis='columns', level='variable').mean()
    show_legend = ( variable_avg_df.columns.size <= 10 )
    variable_avg_df = df.groupby(axis='columns', level='variable').mean()
    col_list = [
        'Average_cost_of_fossil_fuels_in_electricity_generation_per_Btu_dollars_per_million_Btu',
        'Average_retail_price_of_electricity_cents_per_kilowatthour',
        'Consumption_by_electricity_generation_MMBtu',
        'Fossil_fuel_stocks_in_electricity_generation_MMBtu',
        'Net_generation_thousand_megawatthours'
    ]
    variable_avg_df = variable_avg_df[col_list]
    denom_row = variable_avg_df.iloc[0]
    variable_avg_df = variable_avg_df.apply(lambda x: x / denom_row, axis='columns').head(100)
    variable_avg_df.plot(title='% Change Over Time', legend=show_legend, figsize=(9, 6))
    # Sort variable list to start with target variable.
    variables = df.columns.get_level_values('variable').unique().to_list()
    var_list = [target_var, 'Retail_sales_of_electricity_million_kilowatthours',
                'Revenue_from_retail_sales_of_electricity_million_dollars',
                'Receipts_of_fossil_fuels_electricity_generation_MMBtu']
    for var in var_list:
        if var in variables:
            variables.remove(var)
    graph_var_list = [target_var] + variables
    # Replace zeros with NaNs for better visualization.
    df = df.replace(0, np.nan)
    discete_feat_ct = df.columns.nlevels - 1
    graph_rows = len(graph_var_list)
    # Plot discrete feature effects on variables by column hierarchy.
    fig, axs = plt.subplots(figsize=(5 * discete_feat_ct, 4 * graph_rows), 
                              nrows=graph_rows, ncols=discete_feat_ct,
                              sharex='none', sharey='none') # sharex='col', sharey='row'
    for row_num in range(len(graph_var_list)):
        df_one_var = df[graph_var_list[row_num]]
        for col_num in range(df_one_var.columns.nlevels):
            col_lvl_name = df_one_var.columns.names[col_num]
            col_lvl_df = df_one_var.groupby(axis='columns', level=col_lvl_name).mean()
            show_legend = ( col_lvl_df.columns.size <= 10 )
            col_lvl_df.plot(title=graph_var_list[row_num][:30]+'(by '+col_lvl_name+')', 
                            legend=show_legend, ax=axs[row_num, col_num])
    plt.tight_layout()
    plt.show()




"""
    df_book['folder_df_nz_long_sct_conv'].copy(deep=True)
        df_long = times_series_to_long_form(df_book['folder_df_nz_xdate_fill'])
"""
def times_series_to_long_form(input_df):
    df = input_df.copy(deep=True)
    df = pd.DataFrame(df.unstack(['year_month'])).rename(columns={0: 'values'}).reset_index()
    return df
# df_fill_long = times_series_to_long_form(folder_df_nz_xdate_fill_drop)




"""
    Convert DataFrame to a long-format suitable for data modeling / machine learning.
"""
def get_model_df(input_df):
    df = input_df.copy(deep=True)
    drop_vars = ['Retail_sales_of_electricity_million_kilowatthours',
                 'Revenue_from_retail_sales_of_electricity_million_dollars']
    df = df[~df['variable'].isin(drop_vars)]
    df2 = pd.DataFrame(df.groupby(by=['variable', 'location', 'energy_type', 'year_month']).mean()) # Removing sector
    # df['energy_type'] = df['energy_type'].replace(['electricity', 'renewable_and_other'], np.nan)
    df3 = df2.unstack(['variable', 'energy_type'])#, fill_value='coal')
    df3.columns = df3.columns.droplevel(0)
    df4 = df3.fillna(method='ffill', axis='index').fillna(method='bfill', axis='index')
    df4.columns = df4.columns.to_flat_index()
    df4.columns = ['__'.join(col).strip() for col in df4.columns.values]
    return df4
# model_df = get_model_df(df)
# model_df = model_df.reset_index().drop(columns='location')
# display(model_df.head())




"""
    Shift MMBtu variable data into millions of MMBTu to equalize the magnitude difference in variables.
"""
def adjust_MMBtu_units(input_df):
    df = input_df.copy(deep=True)
    adjustments = {
        'Consumption_by_electricity_generation_MMBtu__coal': {
            'name': 'Consumption_by_electricity_generation_million_MMBtu__coal',
            'denom': 1000
        },
        'Consumption_by_electricity_generation_MMBtu__natural_gas': {
            'name': 'Consumption_by_electricity_generation_million_MMBtu__natural_gas',
            'denom': 1000
        },
        'Consumption_by_electricity_generation_MMBtu__petroleum_liquids': {
            'name': 'Consumption_by_electricity_generation_million_MMBtu__petroleum_liquids',
            'denom': 1000
        },
        'Consumption_by_electricity_generation_MMBtu__petroleum_coke': {
            'name': 'Consumption_by_electricity_generation_million_MMBtu__petroleum_coke',
            'denom': 1000
        },
        'Fossil_fuel_stocks_in_electricity_generation_MMBtu__coal': {
            'name': 'Fossil_fuel_stocks_in_electricity_generation_million_MMBtu__coal',
            'denom': 1000
        },
        'Fossil_fuel_stocks_in_electricity_generation_MMBtu__coal': {
            'name': 'Fossil_fuel_stocks_in_electricity_generation_million_MMBtu__coal',
            'denom': 1000
        },
        'Fossil_fuel_stocks_in_electricity_generation_MMBtu__petroleum_coke': {
            'name': 'Fossil_fuel_stocks_in_electricity_generation_million_MMBtu__petroleum_coke',
            'denom': 1000
        },
        'Receipts_of_fossil_fuels_electricity_generation_MMBtu__coal': {
            'name': 'Receipts_of_fossil_fuels_electricity_generation_million_MMBtu__coal',
            'denom': 1000
        },
        'Receipts_of_fossil_fuels_electricity_generation_MMBtu__natural_gas': {
            'name': 'Receipts_of_fossil_fuels_electricity_generation_million_MMBtu__natural_gas',
            'denom': 1000
        },
        'Receipts_of_fossil_fuels_electricity_generation_MMBtu__petroleum_liquids': {
            'name': 'Receipts_of_fossil_fuels_electricity_generation_million_MMBtu__petroleum_liquids',
            'denom': 1000
        },
        'Receipts_of_fossil_fuels_electricity_generation_MMBtu__petroleum_coke': {
            'name': 'Receipts_of_fossil_fuels_electricity_generation_million_MMBtu__petroleum_coke',
            'denom': 1000
        }
    }
    for key, var_d in adjustments.items():
        df[key] = df[key] / var_d['denom']
        df = df.rename(columns={key: var_d['name']}) 
    return df
# model_df2 = adjust_MMBtu_units(model_df)




"""
    Generate chronological features.
"""
def add_chrono_features(input_df):
    df = input_df.copy(deep=True)
    df['year'] = df['year_month'].dt.year
    df['year_starting_2008'] = ( df['year'] - df['year'].min() + 1 )
    df['month'] = df['year_month'].dt.month
    # Generate sequential day count to represent smallest time slice.
    date_to_int_mapping = { y:x for x, y
        in dict(enumerate(pd.Series(df['year_month'].sort_values().unique()).sort_values(), 1)).items()
    }
    df['time_period'] = df['year_month'].map(date_to_int_mapping)
    df = df.drop(columns=['year_month'])
    return df
# model_df3 = add_chrono_features(model_df2)




"""
    Viewing of raw feature variable data distributions with kernel-density est.'s
    and a normalized distribution in black.  Helper function to transform_check_distributions().
"""
def show_distributions(df, col_list, max_plt_cols):
    # Replace zeros with NaNs for better visualization.
    df = df.replace(0, np.nan)
    var_ct = len(col_list)
    overflow_rows = var_ct // max_plt_cols + 1
    overflow_cols = min(max_plt_cols, var_ct)
    fig4, axs4 = plt.subplots(figsize=(4 * overflow_cols, 4 * overflow_rows), 
                              nrows=overflow_rows, ncols=overflow_cols,
                              sharex='none', sharey='none')
    for ct in range(0, var_ct):
        c = (ct % max_plt_cols)
        r = (ct // max_plt_cols)
        if overflow_rows > 1:
            g = sns.distplot(df[col_list[ct]].dropna(), kde=True, fit=norm, ax=axs4[r, c])
            axs4[r, c].set_title(col_list[ct][:30], fontdict={'fontsize': 11})
            # axs4[r, c].set_xticklabels(g.get_xticklabels(), rotation=60)
            axs4[r, c].set(xlabel=None, ylabel=None)
        else:
            g = sns.distplot(df[col_list[ct]].dropna(), kde=True, fit=norm, ax=axs4[c])
            axs4[c].set_title(col_list[ct][:30], fontdict={'fontsize': 11})
            axs4[c].set(xlabel=None, ylabel=None)
    plt.tight_layout()
    plt.show()




"""
    Transform selected columns with np.log1p and review distributions for normality appearance.
"""
def transform_check_distributions(input_df, max_plt_cols = 5):
    df = input_df.copy(deep=True)
    transform_cols = [
        'Consumption_by_electricity_generation_million_MMBtu__coal',
        'Consumption_by_electricity_generation_million_MMBtu__natural_gas',
        'Consumption_by_electricity_generation_million_MMBtu__petroleum_liquids',
        'Consumption_by_electricity_generation_million_MMBtu__petroleum_coke',
        'Fossil_fuel_stocks_in_electricity_generation_million_MMBtu__coal',
        'Fossil_fuel_stocks_in_electricity_generation_million_MMBtu__coal',
        'Fossil_fuel_stocks_in_electricity_generation_million_MMBtu__petroleum_coke',
        'Receipts_of_fossil_fuels_electricity_generation_million_MMBtu__coal',
        'Receipts_of_fossil_fuels_electricity_generation_million_MMBtu__natural_gas',
        'Receipts_of_fossil_fuels_electricity_generation_million_MMBtu__petroleum_liquids',
        'Receipts_of_fossil_fuels_electricity_generation_million_MMBtu__petroleum_coke',
        'Net_generation_thousand_megawatthours__coal',
        'Net_generation_thousand_megawatthours__natural_gas',
        'Net_generation_thousand_megawatthours__petroleum_liquids',
        'Net_generation_thousand_megawatthours__renewable_and_other',
        'Net_generation_thousand_megawatthours__petroleum_coke'
    ]
    df[transform_cols] = df[transform_cols].apply(np.log1p)
    df = df.replace([np.inf, -np.inf], 0).sort_values(by=df.columns.to_list())
    df = df.fillna(method='bfill', axis='index').fillna(method='ffill', axis='index')
    distribution_cols = df.columns.to_list()
    distribution_cols.remove('year')
    distribution_cols.remove('time_period')
    show_distributions(df, distribution_cols, max_plt_cols)
    return df
# model_df4 = transform_check_distributions(model_df3)
# model_df4 = model_df4.reset_index(drop=True)
# target_var = 'Average_retail_price_of_electricity_cents_per_kilowatthour__electricity'




"""
    Plot Effects on Continuous Target Variables Broken Down By Discrete Feature.
"""
def discrete_feature_effects(starting_df, target_var, discrete_feature_cols):
    # Replace zeros with NaNs for better visualization.
    df = starting_df.replace(0, np.nan).copy(deep=True)
    discete_feat_ct = len(discrete_feature_cols)
    graph_rows = 3
    fig, axs = plt.subplots(figsize=(6 * discete_feat_ct, 4.5 * graph_rows), 
                              nrows=graph_rows, ncols=discete_feat_ct,
                              sharex='none', sharey='none') # sharex='col', sharey='row'
    for c in range(0, discete_feat_ct):
        if df[discrete_feature_cols[c]].nunique() <= 30:
            # Create a grouped DataFrame for the feature.
            discrete_feature_df = df.groupby(discrete_feature_cols[c])[target_var]
            discrete_feature_count_df = df.fillna("NULL FEATURE").groupby(discrete_feature_cols[c])\
                                        .agg(['count'])[target_var].sort_values('count', ascending=True)
            # Graph categorized kernel density estimates of distributions of feature effects on target.
            ax1 = discrete_feature_df.apply(
                lambda x: sns.distplot(x, hist=False, kde=True, rug=False, label=x.name, ax=axs[0, c]))
            axs[0, c].set_title(target_var[:30] + ' (by ' + discrete_feature_cols[c] + ')', fontdict={'fontsize': 11})
            # Graph categorized boxplots of feature effects on target.
            x_order = discrete_feature_df.median().sort_values(ascending=True).index
            ax2 = sns.boxplot(x=discrete_feature_cols[c], y=target_var, data=df, order=x_order, ax=axs[1, c])
            axs[1, c].set_title(target_var[:30] + ' (by ' + discrete_feature_cols[c] + ')', fontdict={'fontsize': 11})
            axs[1, c].set_xticklabels(ax2.get_xticklabels(), rotation=35)
            axs[1, c].set_ylabel('')
            axs[1, c].set_xlabel('')
            # Graph categorized horizontal bars to see if category counts are even or uneven.
            ax3 = sns.barplot(x='count', y=discrete_feature_count_df.index.name,
                              data=discrete_feature_count_df.reset_index(), ax=axs[2, c])
    plt.tight_layout()
    plt.show()
# for var in graph_var_list:
#     target_var_graph_df = graph_df[var].reset_index().dropna()
#     discrete_feature_effects(target_var_graph_df.reset_index(),
#                              var, ['energy_type', 'sector'])




"""
    Split data into a training set and test set.
"""
def split_training_test_data(model_data_df, target_var, test_size_arg, random_state_arg):
    # Separate target and feature variables.
    Y = model_data_df[target_var]
    X = model_data_df.drop(columns=[target_var])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, Y, test_size=test_size_arg, random_state=random_state_arg)
    print("The number of observations in training set is {}".format(X_train.shape[0]))
    print("The number of observations in test set is {}".format(X_test.shape[0]))
    return X_train, X_test, y_train, y_test
# test_size = 0.20
# random_state = 432
# X_train, X_test, y_train, y_test = \
#     split_training_test_data(model_df4.copy(deep=True), target_var, test_size, random_state)




"""
    Generate a Correlation Matrix.
    Helper function for generate_linear_regression().
"""
def show_correlations(input_df, target_var_input, fig_size=(11, 11), selected_cols=None):
    if selected_cols == None:
        selected_cols = input_df.columns.tolist()
    df = input_df[selected_cols].copy(deep=True)
    rename_col_dict = { 
        long_name: long_name[:22] + "..." + long_name.split("__")[-1] \
            for long_name in df.columns.to_list() \
            if long_name.__contains__("__")
    }
    df = df.rename(columns=rename_col_dict)
    target_var = target_var_input[:22] + "..." + target_var_input.split("__")[-1] 
    corr_matrix = df.corr()\
                  .sort_values(by=target_var, ascending=False, axis='columns')\
                  .sort_values(by=target_var, ascending=False, axis='index')
    plt.figure(figsize=fig_size)
    g5 = sns.heatmap(corr_matrix.round(2), square=True, annot=True, linewidths=0.5)
    plt.tight_layout()
    plt.show()




"""
    Generate a linear regression.
"""
def generate_linear_regression(model_data_df, target_var, X_train, y_train, X_test, y_test):
    
    # Stats Model OLS Linear Regression needs a manually-added constant.
    X_train_sm = sm.add_constant(X_train)
    
    # Run OLS linear regression with Stats Model.
    results = sm.OLS(y_train, X_train_sm).fit()
    print("Stats Models Results:")
    display(results.summary())
    print("The p-values are less than zero for all coefficients, so they are statistically significant.")
    
    # Stats Model OLS Prediction.
    X_test_sm = sm.add_constant(X_test)
    y_predictions = results.predict(X_test_sm)
    plt.scatter(y_test, y_predictions)
    plt.plot(y_test, y_test, color="red")
    plt.xlabel("true values")
    plt.ylabel("predicted values")
    plt.title("Charges: true and predicted values")
    plt.show()

    print("Mean absolute error of the prediction is: {:,.1f}".format(mean_absolute_error(y_test, y_predictions)))
    print("Mean squared error of the prediction is: {:,.1f}".format(mse(y_test, y_predictions)))
    print("Root mean squared error of the prediction is: {:,.1f}".format(rmse(y_test, y_predictions)))
    print("Mean absolute percentage error of the prediction is: {:,.1f}".format(
        np.mean(np.abs((y_test - y_predictions) / y_test)) * 100))
    
    # Create a LinearRegression model object from scikit-learn's linear_model module.
    lrm = LinearRegression()
    
    # Fit method estimates the coefficients using OLS.
    lrm.fit(X_train, y_train)
    
    # Inspect the results.
    print('\nCoefficients: \n', lrm.coef_)
    print('\nIntercept: \n', lrm.intercept_)
    predictions = lrm.predict(X_train)
    
    r_score_test = lrm.score(X_test, y_test)

    # Assumption two: The error term should be zero on average.
    errors = y_train - predictions
    print("The error term should be zero on average.")
    print("Mean of the errors in the model is: {:,.3f}".format(np.mean(errors)))

    # Assumption three: Homoscedasticity.
    plt.scatter(predictions, errors)
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.axhline(y=0)
    plt.title('Residual vs. Predicted')
    plt.show()
    bart_stats = bartlett(predictions, errors)
    lev_stats = levene(predictions, errors)
    print("Bartlett test statistic value is {0:,.1f} and p value is {1:,.3f}".format(bart_stats[0], bart_stats[1]))
    print("Levene test statistic value is {0:,.1f} and p value is {1:,.3f}".format(lev_stats[0], lev_stats[1]))
    print("Bartlett and Levene tests both share a null hypothesis that the errors are homoscedastic.",
          "If the p-values are less than 0.05, then the results reject the null hypothesis and the errors are",
          "heteroscedastic.")
    print("Causes of heteroscedasticity include outliers in the data and omitted variables important in explaining",
          "the target variance. Include relevant features that target the poorly-estimated areas or transform the",
          "dependent variable. Models which suffer from heteroscedasticity still have estimated coefficients which",
          "are consistent (still valid).  The reliability of some statistical tests, like the t-test, are affected",
          "and may make some estimated coefficients falsely appear to be statistically insignificant.")

    # Assumption four: Low multicollinearity.
    show_correlations(pd.concat([X_train, y_train], axis='columns')\
                      .drop(columns=['year', 'time_period']), y_train.name, (14, 14))
    
    print("Individual features are only weakly correlated with one another, therefore we have low multicolinearity.")

    # Assumption five: The error terms should be uncorrelated with one another.
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.scatter(y_train, errors)
    plt.subplot(1,2,2)
    acf_data = acf(errors)
    plt.plot(acf_data[1:])
    plt.show()
    print("The error terms should be uncorrelated with one another (low R-values).")

    # Assumption six: Exogeneity, that features shouldn't be correlated with the errors.
    rand_nums = np.random.normal(np.mean(errors), np.std(errors), len(errors))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.scatter(np.sort(rand_nums), np.sort(errors)) # we sort the arrays
    plt.xlabel("the normally distributed random variable")
    plt.ylabel("errors of the model")
    plt.title("QQ plot")
    plt.subplot(1,2,2)
    plt.hist(errors)
    plt.xlabel("errors")
    plt.title("Histogram of the errors")
    plt.tight_layout()
    plt.show()
    jb_stats = jarque_bera(errors)
    norm_stats = normaltest(errors)
    print("Jarque-Bera test statistics is {0:,.1f} and p value is {1:,.3f}".format(jb_stats[0], jb_stats[1]))
    print("Normality test statistics is {0:,.1f} and p value is {1:,.3f}".format(norm_stats[0], norm_stats[1]))
    print("The errors appear to be normally distributed from a visual inspection.")
    print("The p-values of both tests (<0.05) indicate that our errors are not normally distributed.")
    
    # We add constant to the model as it's a best practice
    # to do so every time!
    X_test = sm.add_constant(X_test)

    # We are making predictions here
    y_preds = results.predict(X_test)

    plt.scatter(y_test, y_preds)
    plt.plot(y_test, y_test, color="red")
    plt.xlabel("true values")
    plt.ylabel("predicted values")
    plt.title("Charges: true and predicted values")
    plt.show()

    print("Mean absolute error of the prediction is: {:,.1f}".format(mean_absolute_error(y_test, y_preds)))
    print("Mean squared error of the prediction is: {:,.1f}".format(mse(y_test, y_preds)))
    print("Root mean squared error of the prediction is: {:,.1f}".format(rmse(y_test, y_preds)))
    print("Mean absolute percentage error of the prediction is: {:,.1f}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))
    
    return [r_score_test, rmse(y_test, y_predictions)]
# # Generate linear regression using OLS and check for Markov assumptions.
# lr_rval, lr_rmse = generate_linear_regression(model_df4, target_var, X_train, y_train, X_test, y_test)
# print("SK Learn Linear Regression - Adjusted R-squared value: {:,.2f} with RMSE of {:,.1f}.".format(
#     lr_rval, lr_rmse))



