# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:49:47 2021

@author: prite
"""

import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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


# Impute missing data for missingness below a threshold
def impute_missing_data(df, thresh):
    imputer = IterativeImputer()
    low_missing_cols = df.columns[df.isnull().mean() < thresh]
    high_missing_cols = df.columns[df.isnull().mean() > thresh]
    for cols in low_missing_cols:
        df[cols] = imputer.fit_transform(df[[cols]])

    return df, high_missing_cols


# Remove negative values for specified logs
def remove_negatives(df, cols):
    df[df[cols] < 0] = 0

    return df


# Create lagged features and rolling mean window for logs
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


# Convert resistivitiy log to log scale
def convert_res_to_log(df):
    df["RESM"] = np.log10(df["RESM"])
    df["RESD"] = np.log10(df["RESD"])

    return df


# Normalize dataset
def normalize_cols(df):
    cols = df.columns
    scaler = MinMaxScaler()
    scaler_fit = scaler.fit(df)
    scaler_transformed = scaler.transform(df)
    normalized_data = pd.DataFrame(scaler_transformed, columns=cols)
    
    return normalized_data, scaler_fit  # , scaler_transformed


# Apply minmax scaler
def apply_minmaxscaler(df, cols):
    scaler = MinMaxScaler()
    scaler_fit = scaler.fit(df)
    scaler_transformed = scaler.transform(df)
    scaler_transformed = pd.DataFrame(scaler_transformed, columns=cols)

    return scaler_transformed, scaler_fit


# Normalize test data
def normalize_test(df, scaler):
    cols = df.columns
    normalized_data = scaler.transform(df)
    normalized_data = pd.DataFrame(normalized_data, columns=cols)

    return normalized_data

# Power transform columns
def powertransform_cols(df):
    cols = df.columns
    scaler = PowerTransformer(method="yeo-johnson")
    ct = ColumnTransformer([("transform", scaler, cols)], remainder="passthrough")
    PowerTransformer_data = ct.fit_transform(df)
    PowerTransformer_data = pd.DataFrame(PowerTransformer_data, columns=cols)

    return PowerTransformer_data, scaler


# Inverse transform normalized columns
def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values


# Remove outliers from dataset
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

    print("Number of points before outliers removed                     :", len(df))
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

    plt.figure(figsize=(13,10))

    plt.subplot(3,2,1)
    df.boxplot()
    plt.title('Before Outlier Removal', size=15)

    plt.subplot(3,2,2)
    well_train_std.boxplot()
    plt.title('After Outlier Removal with Standard Deviation Filter', size=15)

    plt.subplot(3,2,3)
    well_train_iso.boxplot()
    plt.title('After Outlier Removal with Isolation Forest', size=15)

    plt.subplot(3,2,4)
    well_train_ee.boxplot()
    plt.title('After Outlier Removal with Min. Covariance', size=15)

    plt.subplot(3,2,5)
    well_train_lof.boxplot()
    plt.title('After Outlier Removal with Local Outlier Factor', size=15)

    plt.subplot(3,2,6)
    well_train_svm.boxplot()
    plt.title('After Outlier Removal with One-class SVM', size=15)

    plt.tight_layout(1.7)
    plt.show()

    return well_train_svm


# Plot well logs
def log_plot(logs, well):
    logs = logs.sort_values(by="DEPTH")
    top = logs["DEPTH"].min()
    bot = logs["DEPTH"].max()

    f, ax = plt.subplots(nrows=1, ncols=6, figsize=(12, 8))
    ax[0].plot(logs.GR, logs["DEPTH"], color="green")
    ax[1].plot(logs.RESD, logs["DEPTH"], color="red")
    ax[2].plot(logs.NPHI, logs["DEPTH"], color="black")
    ax[3].plot(logs.RHOB, logs["DEPTH"], color="c")
    ax[4].plot(logs.DTCO, logs["DEPTH"], color="blue")
    ax[5].plot(logs.DTSM, logs["DEPTH"], color="m")

    for i in range(len(ax)):
        ax[i].set_ylim(top, bot)
        ax[i].invert_yaxis()
        ax[i].grid()

    ax[0].set_xlabel("GR")
    ax[0].set_xlim(0, logs.GR.max())
    ax[0].set_ylabel("Depth(ft)")
    ax[1].set_xlabel("RESD")
    ax[1].set_xlim(0, 4)
    ax[2].set_xlabel("NPHI")
    ax[2].set_xlim(-0.4, 0.6)
    ax[3].set_xlabel("RHOB")
    ax[3].set_xlim(logs.RHOB.min(), logs.RHOB.max())
    ax[4].set_xlabel("DTCO")
    ax[4].set_xlim(logs.DTCO.min(), logs.DTCO.max())
    ax[5].set_xlabel("DTSM")
    ax[5].set_xlim(logs.DTSM.min(), logs.DTSM.max())
    

    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_yticklabels([])
    ax[5].set_yticklabels([])

    f.suptitle("Well: " + well, fontsize=14, y=0.94)
