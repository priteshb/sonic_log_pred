# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:49:47 2021

@author: prite
"""

# import config as cnfg
# import lasio
# import os
import collections
# import pickle
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
# from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# import xgboost as xg

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


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


def impute_missing_data(df, thresh):
    imputer = IterativeImputer()
    low_missing_cols = df.columns[df.isnull().mean() < thresh]
    high_missing_cols = df.columns[df.isnull().mean() > thresh]
    for cols in low_missing_cols:
        df[cols] = imputer.fit_transform(df[[cols]])

    return df, high_missing_cols


def remove_negatives(df, cols):
    df[df[cols] < 0] = 0

    return df


def create_lag_features(df, param, lags=None, wins=None):
    lag_cols = [param + "_lag_" + str(lag) for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df[param].shift(2 * lag)
        # print(f"created lag {lag} for {param}")

    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            df[param + "_rmean_" + str(lag) + "_" + str(win)] = df[lag_col].transform(
                lambda x: x.rolling(2 * win).mean()
            )
            # print(f"created rolling win {win} for lag {lag} for {param}")

    print(df.shape)

    return df


def convert_res_to_log(df):
    df["RESM"] = np.log10(df["RESM"])
    df["RESD"] = np.log10(df["RESD"])

    return df


def normalize_cols(df):
    # normalize using power transform Yeo-Johnson method
    # scaler = PowerTransformer(method='yeo-johnson')
    cols = df.columns
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    normalized_data = pd.DataFrame(normalized_data, columns=cols)

    ## ColumnTransformer
    # ct = ColumnTransformer([('transform', scaler, pred_vars)], remainder='passthrough')

    # ## fit and transform
    # transformed_data = ct.fit_transform(df)
    # transformed_data = pd.DataFrame(transformed_data, columns=pred_vars)

    return normalized_data, scaler

def normalize_test(df, scalar):
    cols = df.columns
    normalized_data = scalar.fit_transform(df)
    normalized_data = pd.DataFrame(normalized_data, columns=cols)

    return normalized_data


def powertransform_cols(df):
    cols = df.columns
    scaler = PowerTransformer(method="yeo-johnson")
    ct = ColumnTransformer([('transform', scaler, cols)], remainder='passthrough')
    PowerTransformer_data = ct.fit_transform(df)
    PowerTransformer_data = pd.DataFrame(PowerTransformer_data, columns=cols)

    return PowerTransformer_data, scaler


def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values


def outlier_detection(df):
    # Method 1: Standard Deviation Method (traditional)
    well_train_std = df[np.abs(df - df.mean()) <= (3 * df.std())]

    ## delete all rows that have NaNs
    well_train_std = well_train_std.dropna()

    # Method 2: Isolation Forest
    iso = IsolationForest(contamination=0.5)
    yhat = iso.fit_predict(df)
    mask = yhat != -1
    well_train_iso = df[mask]

    # Method 3: Minimum Covariance Determinant
    ee = EllipticEnvelope(contamination=0.1)
    yhat = ee.fit_predict(df)
    mask = yhat != -1
    well_train_ee = df[mask]

    # Method 4: Local Outlier Factor
    lof = LocalOutlierFactor(contamination=0.3)
    yhat = lof.fit_predict(df)
    mask = yhat != -1
    well_train_lof = df[mask]

    # Method 5: One-class SVM
    svm = OneClassSVM(nu=0.1)
    yhat = svm.fit_predict(df)
    mask = yhat != -1
    well_train_svm = df[mask]

    print("Number of points before outliers removed                     :", 
          len(df)
          )
    print(
        "Number of points after outliers removed with Standard Deviation:",
        len(well_train_std),
    )
    print(
        "Number of points after outliers removed with Isolation Forest  :",
        len(well_train_iso),
    )
    print(
        "Number of points after outliers removed with Min. Covariance   :",
        len(well_train_ee),
    )
    print(
        "Number of points after outliers removed with Outlier Factor    :",
        len(well_train_lof),
    )
    print(
        "Number of points after outliers removed with One-class SVM     :",
        len(well_train_svm),
    )

    # plt.figure(figsize=(13,10))

    # plt.subplot(3,2,1)
    # df[vars_to_use].boxplot()
    # plt.title('Before Outlier Removal', size=15)

    # plt.subplot(3,2,2)
    # well_train_std[vars_to_use].boxplot()
    # plt.title('After Outlier Removal with Standard Deviation Filter', size=15)

    # plt.subplot(3,2,3)
    # well_train_iso[vars_to_use].boxplot()
    # plt.title('After Outlier Removal with Isolation Forest', size=15)

    # plt.subplot(3,2,4)
    # well_train_ee[vars_to_use].boxplot()
    # plt.title('After Outlier Removal with Min. Covariance', size=15)

    # plt.subplot(3,2,5)
    # well_train_lof[vars_to_use].boxplot()
    # plt.title('After Outlier Removal with Local Outlier Factor', size=15)

    # plt.subplot(3,2,6)
    # well_train_svm[vars_to_use].boxplot()
    # plt.title('After Outlier Removal with One-class SVM', size=15)

    # plt.tight_layout(1.7)
    # plt.show()

    return well_train_svm