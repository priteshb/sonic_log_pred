# -*- coding: utf-8 -*-
"""
SPE GCS Competition
"""

import config as cnfg
import utils as ut
import lasio
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xg


pd.set_option("max_columns", None)

file_list = []
file_list += [file for file in os.listdir(os.curdir + '/test_data') 
              if file.endswith(".las")]

scalar_fn = cnfg.scalar_fn
model_fn = cnfg.model_fn

missing_value = cnfg.missing_value
missingness_thresh = cnfg.missingness_thresh
vars_to_use = cnfg.vars_to_use
response_var = cnfg.response_var
pred_vars = cnfg.pred_vars
nonneg_vars = cnfg.nonneg_vars
nonneg_vars = [x for x in nonneg_vars if x != response_var]
thresh = cnfg.thresh
res_lags = cnfg.res_lags
gr_lags = cnfg.gr_lags
nphi_lags = cnfg.nphi_lags
rhob_lags = cnfg.rhob_lags
dtco_lags = cnfg.dtco_lags
res_win = cnfg.res_win
gr_win = cnfg.gr_win
nphi_win = cnfg.nphi_win
dtco_win = cnfg.dtco_win
rhob_win = cnfg.rhob_win

log_mapping = pd.read_excel("Logs_Mapping.xlsx", sheet_name="Distinct mnemonics")


test_df = pd.DataFrame()
test_all = {}
inputlas = {}

# Read files and get log stats in dataframe
for file in file_list:
    # file = '0e121cce5c23_TGS.las'
    # file = '1cf78b7ca1cc_TGS.las'
    # file = '4bc281e7f645_TGS.las'
    # file = '70a049901d0c_TGS.las'
    inputlas[file] = lasio.read('./test_data/'+file)  # Read file
    print(file)

    df = inputlas[file].df()  # Convert data to dataframe
    df = df.rename_axis("DEPT").reset_index()  # Create depth axis and reset index
    df = df.replace(missing_value, "")  # Convert missing value validationt o null
    # df = df.dropna(subset=["DTSM"])
    # des = pd.DataFrame(df.describe())  # Get data stats

    # for curves in inputlas.curves:
    #     # if 'SFL' in curves.mnemonic:
    #     # print(file)
    #     # if curves.mnemonic not in mnemonics:
    #         # print(curves.mnemonic)
    #     curv_desc = [file, curves.mnemonic, curves.unit, curves.descr]
    #     curv_stats = list(des.loc[:, curves.mnemonic].values)
    #     missingness = 100*df[curves.mnemonic].isnull().mean()
    #     curv_desc.extend(curv_stats)
    #     curv_desc.extend([missingness])
    #     temp_df = pd.DataFrame([curv_desc],
    #                            columns=['FILE', 'LOG', 'UNIT', 'DESC', 'COUNT',
    #                                     'MEAN', 'STD', 'MIN', '25%', '50%',
    #                                     '75%', 'MAX', 'MISSINGNESS'])
    #     temp_df = temp_df[temp_df['COUNT'] > 0]
    #     mnemonics_df = mnemonics_df.append(temp_df)

    # df = df.dropna(axis=1, how="all")
    df = ut.log_renaming_shortlisting(df, log_mapping, 'DTCO')
    # print(df.isna().mean())
    print(df.columns)
    
    if not all(x in df.columns for x in pred_vars):
        for col in pred_vars:
            if col not in df.columns:
                df[col] = np.nan
    df = df[pred_vars]
    df, high_missing_cols = ut.impute_missing_data(df, missingness_thresh)
    # if len(high_missing_cols) > 0:
    #     df = df.dropna(axis=1, how="any")
    # if len(df.columns) == len(pred_vars):
        # df = df[df["RESD"] > 0] #remove negative resistivities
        # df = df[df["RESM"] > 0]
    df[df[nonneg_vars] < 0] = 0 # remove negative values 
    # df = ut.outlier_detection(df)
    # df = ut.convert_res_to_log(df)
    # df = ut.normalize_cols(df)
    # df = ut.outlier_detection(df)

    # sns.pairplot(df, vars=vars_to_use, diag_kind='kde',
    #              plot_kws = {'alpha': 0.6, 's': 30, 'edgecolor': 'k'})
    # df, scaler = normalize_cols(df)

    df = ut.create_lag_features(df, "RESD", lags=res_lags, wins=res_win)
    df = ut.create_lag_features(df, "RESM", lags=res_lags, wins=res_win)
    df = ut.create_lag_features(df, "GR", lags=gr_lags, wins=gr_win)
    df = ut.create_lag_features(df, "NPHI", lags=nphi_lags, wins=nphi_win)
    df = ut.create_lag_features(df, "RHOB", lags=rhob_lags, wins=rhob_win)
    df = ut.create_lag_features(df, "DTCO", lags=dtco_lags, wins=dtco_win)
    # df = df.fillna(method='ffill')
    # df = df.fillna(method='bfill')
    # df = df.dropna(axis=0, how="any")

    # print(f"Appending {file} to main df")
    test_all[file] = df
    print(df.shape)
     

scalar_x = joblib.load('scaler_x.pkl')
scalar_y = joblib.load('scaler_y.pkl')

final_dict = {}

for file in file_list:
    print(file)
    test_df_norm = ut.normalize_test(test_all[file], scalar_x)
    print(f'{file} normalized')
    
    #Train and test data division
    test_x = np.asarray(test_df_norm)
    
    # XG Boost
    m_lgb = joblib.load(model_fn)
    print(f'Predicting data for {file}')
    pred = m_lgb.predict(test_x)
    print(f'Predicted data for {file}. Now inverse transforming')
    pred = ut.invTransform(scalar_y, pred, "DTSM", ["DTSM"])
    final_dict[file] = pred
    print('Added output to final_dict')
    well_data = pd.DataFrame(pred, columns=[file])
    well_data.to_excel(file.split('.')[0] + '.xlsx', index = False)
    
    

