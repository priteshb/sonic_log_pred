# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:45:53 2021

@author: prite
"""

#Specify input parameters for training and testing
df_pickle_fn_norm = "df_norm_0131.pkl"
df_pickle_fn = "df_0131.pkl"
df_pickle_fn_pt = "df_pt_0131.pkl"
scalar_fn = 'training_scalar.pkl'
model_fn = 'model.pkl'


missing_value = -9999.25 #Missing value representation
missingness_thresh = 0.2 #Threshold for column missingness in pu

#List of mnemonics to be used for modelling including response log
vars_to_use = ["RESD", "RESM", "DTCO", "DTSM", "NPHI", "RHOB", "GR"]

response_var = "DTSM" #Response log mnemonic

#Predictor logs
pred_vars = [x for x in vars_to_use if x != response_var]

#Logs that cannot have negative values
nonneg_vars = ["RESD", "RESM", "DTCO", "DTSM", "RHOB", "GR"]

#Lagging window length in ft.
res_lags = [4, 6, 8]
gr_lags = [1, 2, 3, 4, 6]
nphi_lags = [1, 3, 5]
rhob_lags = [2, 3, 5]
dtco_lags = [3, 4, 5, 6, 7]

#Rolling mean window length in ft.
res_win = [4, 6]
gr_win = [3, 5]
nphi_win = [4, 6]
dtco_win = [4, 6, 8]
rhob_win = [3, 5]