# -*- coding: utf-8 -*-
"""
SPE GCS Competition
"""

import lasio, os, collections, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import xgboost as xg

pd.set_option("max_columns", None)

file_list = []
file_list += [file for file in os.listdir(os.curdir) if file.endswith(".las")]
df_pickle_fn = "df.pkl"


missing_value = [-9999.25]
mnemonics = []
response_var = "DTSM"
pred_var = ["RESD", "RESM", "DTCO", "DTSM", "NPHI", "RHOB", "GR"]
nonneg_vars = ["RESD", "RESM", "DTCO", "DTSM", "RHOB", "GR"]
thresh = 0.2
res_lags = [3, 4, 5]  # in ft
gr_lags = [2, 3, 4]
nphi_lags = [3, 4, 5]
rhob_lags = [2, 3]
dtco_lags = [5, 6, 7]
res_win = [4]
gr_win = [3]
nphi_win = [4]
dtco_win = [6]
rhob_win = [3]

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


# Identify the best log in a dict
def find_best_logs(df, log_dict, response_var):
    for k, v in log_dict.items():
        if len(v) > 1:
            corr = df[v].corrwith(df[response_var])
            best_log = corr.idxmax()
            log_dict[k] = [best_log]

    return log_dict


# Identify which logs have multiple mappings in a dict
def identify_duplicate_log(log_dict):
    ret_dict = collections.defaultdict(list)
    for k, v in log_dict.items():
        ret_dict[v[0]].append(k)

    return ret_dict


# Rename and shortlist logs
def log_renaming_shortlisting(df, log_map, response_var):
    # df_new = pd.DataFrame()
    col_maps = {}
    try:
        for col in df.columns:
            new_col = log_map.loc[log_map["LOG"] == col, "CATEGORY"].values
            if pd.notna(new_col):
                col_maps[col] = new_col

        col_maps = identify_duplicate_log(col_maps)
        col_maps = find_best_logs(df, col_maps, response_var)
        col_maps = {v[0]: k for k, v in col_maps.items()}
        df = df[df.columns.intersection(list(col_maps.keys()))]
        df = df.rename(columns=col_maps)

    except Exception as e:
        print(e)

    return df


def remove_high_missing_columns(file, df, thresh):
    print(df.isnull().mean())
    return df.columns[df.isnull().mean() < thresh]

def remove_negatives(df, cols):
    print(df.shape)
    df = df[df[cols] > 0]
    print(df.shape)
    return df


def create_lag_features(df, param, lags=None, wins=None):
    lag_cols = [param + "_lag_" + str(lag) for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[param].shift(2 * lag)
        print(f"created lag {lag} for {param}")

    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            df[param + "_rmean_" + str(lag) + "_" + str(win)] = df[lag_col].transform(
                lambda x: x.rolling(2 * win).mean()
            )
            print(f"created rolling win {win} for lag {lag} for {param}")

    print(df.shape)

    return df


mnemonics_df = pd.DataFrame(
    columns=[
        "FILE",
        "LOG",
        "UNIT",
        "DESC",
        "COUNT",
        "MEAN",
        "STD",
        "MIN",
        "25%",
        "50%",
        "75%",
        "MAX",
        "MISSINGNESS",
    ]
)
train_df = pd.DataFrame()
inputlas = {}


# Read files and get log stats in dataframe
for file in file_list:
    # file = '1a000e7f474b_TGS.las'
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
    df = log_renaming_shortlisting(df, log_mapping, response_var)
    if all(x in df.columns for x in pred_var):
        cols_low_missingness = remove_high_missing_columns(file, df, 0.2)

        df = df[pred_var]
        df = remove_negatives(df, nonneg_vars)
        df = create_lag_features(df, "RESD", lags=res_lags, wins=res_win)
        df = create_lag_features(df, "RESM", lags=res_lags, wins=res_win)
        df = create_lag_features(df, "GR", lags=gr_lags, wins=gr_win)
        df = create_lag_features(df, "NPHI", lags=nphi_lags, wins=nphi_win)
        df = create_lag_features(df, "RHOB", lags=rhob_lags, wins=rhob_win)
        df = create_lag_features(df, "DTCO", lags=dtco_lags, wins=dtco_win)

        print(f"Appending {file} to main df")
        train_df = train_df.append(df)

train_df.to_pickle(df_pickle_fn)

x = np.asarray(train_df.loc[:, train_df.columns != response_var])
y = np.asarray(train_df[response_var])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
xgb_r = xg.XGBRegressor(objective="reg:squarederror", n_estimators=1000)
xgb_r.fit(train_x, train_y)
pred = xgb_r.predict(test_x)
rmse = np.sqrt(MSE(test_y, pred))
rmse
xgb_r.feature_importances_

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
    train_df.loc[:, train_df.columns != response_var], label=train_df[response_var]
)

m_lgb = lgb.train(lgb_params, train_data)


mnemonics_df.to_excel("Mnemonics_wth_file_with_stats_DTSM_length.xlsx")


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
