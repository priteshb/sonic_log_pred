# -*- coding: utf-8 -*-
"""
SPE GCS Competition
"""

import config as cnfg
import utils as ut
import lasio
import os
# import collections
# import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import PowerTransformer
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
import xgboost as xg

# from sklearn.ensemble import IsolationForest
# from sklearn.covariance import EllipticEnvelope
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.svm import OneClassSVM

pd.set_option("max_columns", None)

file_list = []
file_list += [file for file in os.listdir(os.curdir) if file.endswith(".las")]
df_pickle_fn_norm = cnfg.df_pickle_fn_norm
df_pickle_fn = cnfg.df_pickle_fn
df_pickle_fn_pt = cnfg.df_pickle_fn_pt

missing_value = cnfg.missing_value
missingness_thresh = cnfg.missingness_thresh
vars_to_use = cnfg.vars_to_use
response_var = cnfg.response_var
pred_vars = cnfg.pred_vars
nonneg_vars = cnfg.nonneg_vars
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


# Check well relative locations:
# lat = []
# lon = []
# inputlas = {}
# for file in file_list:
#     print(file)
#     inputlas[file] = lasio.read(file) #Read file
#     lat.append(inputlas[file].well['SLAT'].value)
#     lon.append(inputlas[file].well['SLON'].value)

# plt.scatter(x=lon, y=lat)
# plt.show()


# mnemonics_df = pd.DataFrame(
#     columns=[
#         "FILE",
#         "LOG",
#         "UNIT",
#         "DESC",
#         "COUNT",
#         "MEAN",
#         "STD",
#         "MIN",
#         "25%",
#         "50%",
#         "75%",
#         "MAX",
#         "MISSINGNESS",
#     ]
# )

train_df = pd.DataFrame()
inputlas = {}

# Read files and get log stats in dataframe
for file in file_list:
    # file = '0e121cce5c23_TGS.las'
    # file = '1cf78b7ca1cc_TGS.las'
    # file = '4bc281e7f645_TGS.las'
    # file = '70a049901d0c_TGS.las'
    inputlas[file] = lasio.read(file)  # Read file
    print(file)

    df = inputlas[file].df()  # Convert data to dataframe
    df = df.rename_axis("DEPT").reset_index()  # Create depth axis and reset index
    df = df.replace(missing_value, "")  # Convert missing value validationt o null
    df = df.dropna(subset=["DTSM"])
    des = pd.DataFrame(df.describe())  # Get data stats

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

    df = df.dropna(axis=1, how="all")
    df = ut.log_renaming_shortlisting(df, log_mapping, response_var)
    if all(x in df.columns for x in vars_to_use):
        df = df[vars_to_use]
        df, high_missing_cols = ut.impute_missing_data(df, missingness_thresh)
        if len(high_missing_cols) > 0:
            df = df.dropna(axis=1, how="any")
        if len(df.columns) == len(vars_to_use):
            df = df[df["RESD"] > 0] #remove negative resistivities
            df = df[df["RESM"] > 0]
            df[df[nonneg_vars] < 0] = 0 # remove negative values 
            # df = ut.outlier_detection(df)
            df = ut.convert_res_to_log(df)
            # df = ut.normalize_cols(df)
            df = ut.outlier_detection(df)

            # sns.pairplot(df, vars=vars_to_use, diag_kind='kde',
            #              plot_kws = {'alpha': 0.6, 's': 30, 'edgecolor': 'k'})
            # df, scaler = normalize_cols(df)

            df = ut.create_lag_features(df, "RESD", lags=res_lags, wins=res_win)
            df = ut.create_lag_features(df, "RESM", lags=res_lags, wins=res_win)
            df = ut.create_lag_features(df, "GR", lags=gr_lags, wins=gr_win)
            df = ut.create_lag_features(df, "NPHI", lags=nphi_lags, wins=nphi_win)
            df = ut.create_lag_features(df, "RHOB", lags=rhob_lags, wins=rhob_win)
            df = ut.create_lag_features(df, "DTCO", lags=dtco_lags, wins=dtco_win)
            df = df.dropna(axis=0, how="any")

            print(f"Appending {file} to main df")
            train_df = train_df.append(df)
            print(train_df.shape)
     
        
train_df.to_pickle(df_pickle_fn)
train_df_norm, scaler = ut.normalize_cols(train_df)
# train_df_norm, scaler = ut.powertransform_cols(train_df)
# train_df_norm.to_pickle(df_pickle_fn_pt)
# train_df_norm = train_df
train_df_norm.to_pickle(df_pickle_fn_norm)

# train_df_norm = pd.read_pickle(df_pickle_fn)

#Train and test data division
x = np.asarray(train_df_norm.loc[:, train_df_norm.columns != response_var])
y = np.asarray(train_df_norm[response_var])

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.3, random_state=11
)


# XG Boost
xgb_r = xg.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_r.fit(train_x, train_y)
pred = xgb_r.predict(test_x)

test_y = ut.invTransform(scaler, test_y, "DTSM", train_df_norm.columns)
pred = ut.invTransform(scaler, pred, "DTSM", train_df_norm.columns)

rmse = np.sqrt(MSE(test_y, pred))
rmse
xgb_r.feature_importances_

# Light GBM
import lightgbm as lgb

lgb_params = {
    #     'nfold': 5,
    "boosting_type": "gbdt",
    "metric": "rmse",
    #     'objective': 'regression',
    "objective": "poisson",
    #     'objective': 'tweedie',
    #     'tweedie_variance_power': 1.1,
    "force_row_wise": True,
    "n_jobs": -1,
    "seed": 236,
    "learning_rate": 0.075,
    "feature_fraction": 0.5,
    #     "sub_feature" : 0.8,
    #     "sub_row" : 0.75,
    "lambda_l2": 0.1,
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    # "colsample_bytree": 0.75,
    "num_leaves": 2 ** 7 - 1,
    "min_data_in_leaf": 2 ** 8 - 1,
    "verbosity": 1,
    "num_boost_round": 600,
    #     'num_iterations' : 1200,
    "n_estimators": 2000,
    # 'device': 'gpu',
    # 'gpu_platform_id': 2,
    # 'gpu_device_id': 1
}

train_data = lgb.Dataset(
    train_df.loc[:, train_df_norm.columns != response_var], label=train_df[response_var]
)

m_lgb = lgb.train(lgb_params, train_data)


# mnemonics_df.to_excel("Mnemonics_wth_file_with_stats_DTSM_length.xlsx")


# def log_plot(logs):
#     logs = logs.sort_values(by="DEPT")
#     top = logs["DEPT"].min()
#     bot = logs["DEPT"].max()

#     f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 8))
#     ax[0].plot(logs.GRR, logs["DEPT"], color="green")
#     ax[1].plot(logs.TNPH, logs["DEPT"], color="red")
#     ax[2].plot(logs.HLLD, logs["DEPT"], color="black")
#     ax[3].plot(logs.RHOZ, logs["DEPT"], color="c")
#     ax[4].plot(logs.VPVS, logs["DEPT"], color="m")

#     for i in range(len(ax)):
#         ax[i].set_ylim(top, bot)
#         ax[i].invert_yaxis()
#         ax[i].grid()

#     ax[0].set_xlabel("GR")
#     ax[0].set_xlim(logs.GRR.min(), logs.GRR.max())
#     ax[0].set_ylabel("Depth(ft)")
#     ax[1].set_xlabel("POR")
#     ax[1].set_xlim(logs.TNPH.min(), logs.TNPH.max())
#     ax[2].set_xlabel("HLLD")
#     ax[2].set_xlim(logs.HLLD.min(), logs.HLLD.max())
#     ax[3].set_xlabel("RHOB")
#     ax[3].set_xlim(logs.RHOZ.min(), logs.RHOZ.max())
#     ax[4].set_xlabel("DTSM")
#     ax[4].set_xlim(logs.VPVS.min(), logs.VPVS.max())

#     ax[1].set_yticklabels([])
#     ax[2].set_yticklabels([])
#     ax[3].set_yticklabels([])
#     ax[4].set_yticklabels([])
#     # ax[5].set_yticklabels([]);
#     # ax[6].set_yticklabels([])

#     f.suptitle("Well: #" + las_filename, fontsize=14, y=0.94)


# log_plot(df)
