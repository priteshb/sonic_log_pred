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


pd.set_option("max_columns", None)

#Read file names for test data
file_list = []
file_list += [file for file in os.listdir(os.curdir + '/final_test_data') 
              if file.endswith(".las")]

#Read constants from config file
scalar_fn = cnfg.scalar_fn
model_fn = cnfg.model_fn
missing_value = cnfg.missing_value
missingness_thresh = cnfg.missingness_thresh
vars_to_use = cnfg.vars_to_use
response_var = cnfg.response_var
pred_vars = cnfg.pred_vars
nonneg_vars = cnfg.nonneg_vars
nonneg_vars = [x for x in nonneg_vars if x != response_var]
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

#Read Manually mapped log mnemonics
log_mapping = pd.read_excel("Logs_Mapping.xlsx", sheet_name="Distinct mnemonics")

#Read normalized scalar objects
scalar_x = joblib.load('scaler_x.pkl')
scalar_y = joblib.load('scaler_y.pkl')


# Read test files 
for file in file_list:
    inputlas = lasio.read('./final_test_data/'+file)  # Read file
    print(file)

    df = inputlas.df()  #Convert data to dataframe
    df_length = len(df) #Dataframe length
    
    df = df.replace(missing_value, "")  # Convert missing value validation to null

    #Rename log mnemonics to a consistent nomenclature
    df = ut.log_renaming_shortlisting(df, log_mapping, 'DTCO')
    
     #If we have all columns to be used for modelling
    if not all(x in df.columns for x in pred_vars):
        for col in pred_vars:
            if col not in df.columns:
                df[col] = np.nan
                
    df = df[pred_vars]  #Filter logs to be used for modelling
    
    #Impute missing rows for columns with < 20% missingness
    df, high_missing_cols = ut.impute_missing_data(df, missingness_thresh)

    df[df[nonneg_vars] < 0] = 0 # set negative values to zero 
    
    #For resistivity logs
    if all(x in df.columns for x in ['RESD', 'RESM']):
        df[df['RESD'] <= 0] = 0.01 #Change negative resistivities to low value
        df[df['RESM'] <= 0] = 0.01

        df = ut.convert_res_to_log(df) #Convert resistivitiy to log scale
        
        #Create lagged features and rolling mean window for resistivity logs
        #for test data
        df = ut.create_lag_features(df, "RESD", lags=res_lags, wins=res_win)
        df = ut.create_lag_features(df, "RESM", lags=res_lags, wins=res_win)

    #Create lagged features and rolling mean window for specified logs
    #for test data
    df = ut.create_lag_features(df, "GR", lags=gr_lags, wins=gr_win)
    df = ut.create_lag_features(df, "NPHI", lags=nphi_lags, wins=nphi_win)
    df = ut.create_lag_features(df, "RHOB", lags=rhob_lags, wins=rhob_win)
    df = ut.create_lag_features(df, "DTCO", lags=dtco_lags, wins=dtco_win)

    test_df_norm = ut.normalize_test(df, scalar_x) #Normalize logs
    
    test_x = np.asarray(test_df_norm) #Get test data as array
    
    m_lgb = joblib.load(model_fn) #Load trained model object
    
    print(f'Predicting data for {file}')
    pred = m_lgb.predict(test_x) #Predict response log 
    
    print(f'Predicted data for {file}. Now inverse transforming')
    #Inverse transform predicted predicted data
    pred = list(ut.invTransform(scalar_y, pred, "DTSM", ["DTSM"]))
    pred_len = len(pred) #Get predicted data length
    
    if df_length == pred_len: #Check if test data and predicted data length match
        print(f'{file} output count passed')
    
    #Plot logs with predicted output
    df['DTSM'] = pred
    df['DEPTH'] = df.index
    ut.log_plot(df, file.split('.')[0])
    
    #Save output to xlsx files based on specified format
    well_data = pd.DataFrame({'Depth': df.index, 'DTSM': pred})
    well_data.to_excel(os.path.dirname(os.path.realpath(__file__)) + 
                        '\\output_files\\' + file.split('.')[0] + '.xlsx',
                        index=False)
    
    

