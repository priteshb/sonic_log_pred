# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:45:53 2021

@author: prite
"""


df_pickle_fn_norm = "df_norm_0131.pkl"
df_pickle_fn = "df_0131.pkl"
df_pickle_fn_pt = "df_pt_0131.pkl"


missing_value = [-9999.25]
missingness_thresh = 0.2
vars_to_use = ["RESD", "RESM", "DTCO", "DTSM", "NPHI", "RHOB", "GR"]
response_var = "DTSM"
pred_vars = [x for x in vars_to_use if x != response_var]
nonneg_vars = ["DTCO", "DTSM", "RHOB", "GR"]
thresh = 0.2
res_lags = [1, 2, 3, 4, 5, 6]  # in ft
gr_lags = [1, 2, 3, 4]
nphi_lags = [1, 2, 3, 4, 5]
rhob_lags = [1, 2, 3]
dtco_lags = [3, 4, 5, 6, 7]
res_win = [4]
gr_win = [3]
nphi_win = [4]
dtco_win = [6]
rhob_win = [3]