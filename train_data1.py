# -*- coding: utf-8 -*-
"""
SPE GCS Competition
"""

import config as cnfg
import utils as ut
import lasio
import os
import joblib
# import collections
# import pickle
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgb
# import xgboost as xg

pd.set_option("max_columns", None)

file_list = []
file_list += [file for file in os.listdir(os.curdir+'/train_data') 
              if file.endswith(".las")]
df_pickle_fn_norm = cnfg.df_pickle_fn_norm
df_pickle_fn = cnfg.df_pickle_fn
df_pickle_fn_pt = cnfg.df_pickle_fn_pt
scalar_fn = cnfg.scalar_fn
model_fn = cnfg.model_fn

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
test_df = pd.DataFrame()
val_df = pd.DataFrame()
inputlas = {}

# Read files and get log stats in dataframe
for file in file_list:
    # file = '0e121cce5c23_TGS.las'
    # file = '1cf78b7ca1cc_TGS.las'
    # file = '4bc281e7f645_TGS.las'
    # file = '70a049901d0c_TGS.las'
    inputlas[file] = lasio.read('./train_data/'+file)  # Read file
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
            # df = ut.create_lag_features(df, "RESD", lags=res_lags, wins=res_win)
            # df = ut.create_lag_features(df, "RESM", lags=res_lags, wins=res_win)
            # df = ut.create_lag_features(df, "GR", lags=gr_lags, wins=gr_win)
            # df = ut.create_lag_features(df, "NPHI", lags=nphi_lags, wins=nphi_win)
            # df = ut.create_lag_features(df, "RHOB", lags=rhob_lags, wins=rhob_win)
            # df = ut.create_lag_features(df, "DTCO", lags=dtco_lags, wins=dtco_win)
            
            # df = ut.normalize_cols(df)
            
            df_train, df_test = train_test_split(df, test_size = 0.2, random_state=11)
            df_test = ut.create_lag_features(df_test, "RESD", lags=res_lags, wins=res_win)
            df_test = ut.create_lag_features(df_test, "RESM", lags=res_lags, wins=res_win)
            df_test = ut.create_lag_features(df_test, "GR", lags=gr_lags, wins=gr_win)
            df_test = ut.create_lag_features(df_test, "NPHI", lags=nphi_lags, wins=nphi_win)
            df_test = ut.create_lag_features(df_test, "RHOB", lags=rhob_lags, wins=rhob_win)
            df_test = ut.create_lag_features(df_test, "DTCO", lags=dtco_lags, wins=dtco_win)
            
            df_train, df_val = train_test_split(df_train, test_size = 0.1, random_state=11)
            
            df_val = ut.create_lag_features(df_val, "RESD", lags=res_lags, wins=res_win)
            df_val = ut.create_lag_features(df_val, "RESM", lags=res_lags, wins=res_win)
            df_val = ut.create_lag_features(df_val, "GR", lags=gr_lags, wins=gr_win)
            df_val = ut.create_lag_features(df_val, "NPHI", lags=nphi_lags, wins=nphi_win)
            df_val = ut.create_lag_features(df_val, "RHOB", lags=rhob_lags, wins=rhob_win)
            df_val = ut.create_lag_features(df_val, "DTCO", lags=dtco_lags, wins=dtco_win)
            
            df_train = ut.outlier_detection(df_train)
            df_train = ut.create_lag_features(df_train, "RESD", lags=res_lags, wins=res_win)
            df_train = ut.create_lag_features(df_train, "RESM", lags=res_lags, wins=res_win)
            df_train = ut.create_lag_features(df_train, "GR", lags=gr_lags, wins=gr_win)
            df_train = ut.create_lag_features(df_train, "NPHI", lags=nphi_lags, wins=nphi_win)
            df_train = ut.create_lag_features(df_train, "RHOB", lags=rhob_lags, wins=rhob_win)
            df_train = ut.create_lag_features(df_train, "DTCO", lags=dtco_lags, wins=dtco_win)
            # df = df.dropna(axis=0, how="any")

            print(f"Appending {file} to main df")
            # train_df = train_df.append(df)
            train_df = train_df.append(df_train)
            test_df = test_df.append(df_test)
            val_df = val_df.append(df_val)
            print(train_df.shape)
            print(test_df.shape)
            print(val_df.shape)
     
# sns.pairplot(train_df, vars=vars_to_use, diag_kind='kde',
#              plot_kws = {'alpha': 0.6, 's': 30, 'edgecolor': 'k'})

train_df.to_pickle('train_df.pkl')
test_df.to_pickle('test_df.pkl')
val_df.to_pickle('val_df.pkl')
train_df_norm_x, scalar_x = ut.apply_minmaxscaler(train_df.drop([response_var], 
                                                                axis=1), 
                                                  train_df.drop([response_var], 
                                                                axis=1).columns)
train_df_norm_y, scalar_y = ut.apply_minmaxscaler(train_df[[response_var]], 
                                                  [response_var])
joblib.dump(scalar_x, 'scaler_x.pkl') 
joblib.dump(scalar_y, 'scaler_y.pkl') 

train_df_norm = pd.concat([train_df_norm_x, train_df_norm_y], axis=1)
# train_df_norm, scaler = ut.powertransform_cols(train_df)
# train_df_norm.to_pickle(df_pickle_fn_pt)
# train_df_norm = train_df
train_df_norm.to_pickle(df_pickle_fn_norm)

train_x = train_df_norm_x
train_y = train_df_norm_y
test_x = ut.normalize_test(test_df.drop([response_var], axis=1), scalar_x)
test_y = ut.normalize_test(test_df[[response_var]], scalar_y)
# test_df_norm = pd.concat([test_df_norm_x, test_df_norm_y], axis=1)

val_x = ut.normalize_test(val_df.drop([response_var], axis=1), scalar_x)
val_y = ut.normalize_test(val_df[[response_var]], scalar_y)
# val_df_norm = pd.concat([val_df_norm_x, val_df_norm_y], axis=1)

# Light GBM
lgb_params = {
    #     'nfold': 5,
    "boosting_type": "gbdt",
    "metric": "rmse",
    "objective": "regression",
    #     'objective': 'tweedie',
    #     'tweedie_variance_power': 1.1,
    "force_row_wise": True,
    "n_jobs": -1,
    "seed": 11,
    "learning_rate": 0.075,
    "feature_fraction": 0.5,
    #     "sub_feature" : 0.8,
    #     "sub_row" : 0.75,
    "lambda_l2": 0.1,
    "bagging_fraction": 0.75,
    "bagging_freq": 1,
    # "colsample_bytree": 0.75,
    "num_leaves": sp_randint(6, 50),#2 ** 7 - 1,
    "min_child_samples": sp_randint(100, 500),
    "min_data_in_leaf": 30,#2 ** 8 - 1,
    "verbosity": 1,
    "num_boost_round": 600,
    #     'num_iterations' : 1200,
    "n_estimators": 4000,
    # 'device': 'gpu',
    # 'gpu_platform_id': 2,
    # 'gpu_device_id': 1
}

#Train and test data division
# x = np.asarray(train_df_norm.loc[:, train_df_norm.columns != response_var])
# y = np.asarray(train_df_norm[response_var])

# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, 
#                                                     random_state=11)

# train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, 
#                                                     test_size=0.2, 
#                                                     random_state=11)

train_data = lgb.Dataset(train_x, label=train_y)
val_data = lgb.Dataset(val_x, label=val_y)
test_data = lgb.Dataset(test_x, label=test_y)

# m_lgb = lgb.train(lgb_params, train_data)
m_lgb = lgb.train(lgb_params, train_data, early_stopping_rounds=50,
                  valid_sets=[train_data, val_data], verbose_eval=100)

m_lgb_cv = lgb.cv(lgb_params, train_data, nfold=3, stratified=False, 
                  early_stopping_rounds=10)
joblib.dump(m_lgb, model_fn)

pred = m_lgb_cv
pred = m_lgb.predict(test_x, n_jobs=-1)

test_y = ut.invTransform(scalar_y, test_y, "DTSM", train_df.columns)
pred = ut.invTransform(scalar_y, pred, "DTSM", train_df.columns)


rmse = np.sqrt(MSE(test_y, pred))
print(rmse)

def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3


fit_params={"early_stopping_rounds":30, 
            "eval_metric" : 'rmse', 
            "eval_set" : [(val_x, val_y)],
            # 'eval_names': ['valid'],
            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
            'verbose': 100}

param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

# n_HP_points_to_test = 100
clf = lgb.LGBMRegressor(max_depth=-1, random_state=11, silent=True, 
                         metric='rmse', n_jobs=-1, n_estimators=5000)

gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    # n_iter=n_HP_points_to_test,
    scoring='neg_root_mean_squared_error',
    cv=5,
    refit=True,
    random_state=11,
    verbose=True)

gs.fit(train_x.reset_index(drop=True), train_y.reset_index(drop=True), **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

# opt_parameters = {'colsample_bytree': 0.9234, 'min_child_samples': 399, 
#                   'min_child_weight': 0.1, 'num_leaves': 13, 'reg_alpha': 2, 
#                   'reg_lambda': 5, 'subsample': 0.855}

opt_parameters = gs.best_params_

clf_sw = lgb.LGBMRegressor(**clf.get_params())
#set optimal parameters
clf_sw.set_params(**opt_parameters)

gs_sample_weight = GridSearchCV(estimator=clf_sw, 
                                param_grid={'scale_pos_weight':[1,2,6,12]},
                                scoring='neg_root_mean_squared_error',
                                cv=5,
                                refit=True,
                                verbose=True)

gs_sample_weight.fit(train_x.reset_index(drop=True), train_y.reset_index(drop=True), **fit_params)

print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, 
                                                       gs_sample_weight.best_params_))

# print("Valid+-Std     Train  :   Parameters")
# for i in np.argsort(gs_sample_weight.cv_results_['mean_test_score'])[-5:]:
#     print('{1:.3f}+-{3:.3f}     {2:.3f}   :  {0}'.format(gs_sample_weight.cv_results_['params'][i], 
#                                     gs_sample_weight.cv_results_['mean_test_score'][i], 
#                                     gs_sample_weight.cv_results_['mean_train_score'][i],
#                                     gs_sample_weight.cv_results_['std_test_score'][i]))

clf_final = lgb.LGBMRegressor(**clf.get_params())
#set optimal parameters
clf_final.set_params(**opt_parameters)

#Train the final model with learning rate decay
clf_final.fit(train_x, train_y, **fit_params, 
              callbacks=[lgb.reset_parameter(learning_rate=
                                             learning_rate_010_decay_power_0995)])


feat_imp = pd.Series(clf_final.feature_importances_, index=train_x.columns)
feat_imp.nlargest(20).plot(kind='barh', figsize=(8,10))


pred = clf_final.predict(test_x, n_jobs=-1)

test_y_ = ut.invTransform(scalar_y, test_y, "DTSM", train_df.columns)
pred = ut.invTransform(scalar_y, pred, "DTSM", train_df.columns)


rmse = np.sqrt(MSE(test_y_, pred))
print(rmse)


#Bayesian optimization
from bayes_opt import BayesianOptimization

def bayesion_opt_lgbm(dtrain, init_iter=3, n_iters=7, random_state=11, 
                      seed = 101):
  # dtrain = lgb.Dataset(data=X, label=y)

  # Objective Function
  # def hyp_lgbm(num_leaves, feature_fraction, bagging_fraction, max_depth, 
  #              min_split_gain, min_child_weight):
  def hyp_lgbm(params):
          params['application'] = 'regression'
          # params ={'application':'regression','num_iterations': num_iterations,
          #           'learning_rate':0.05, 'early_stopping_round':50,
          #           'metric':'rmse','num_leaves': sp_randint(6, 50), 
          #           'min_child_samples': sp_randint(100, 500), 
          #           'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
          #           'subsample': sp_uniform(loc=0.2, scale=0.8), 
          #           'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
          #           'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
          #           'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
          # # params = {'application':'regression','num_iterations': num_iterations,
          # #           'learning_rate':0.05, 'early_stopping_round':50,
          # #           'metric':'rmse'} # Default parameters
          # params["num_leaves"] = int(round(num_leaves))
          # params['feature_fraction'] = max(min(feature_fraction, 1), 0)
          # params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
          # params['max_depth'] = int(round(max_depth))
          # params['min_split_gain'] = min_split_gain
          # params['min_child_weight'] = min_child_weight
          cv_results = lgb.cv(params, dtrain, nfold=5, seed=seed, stratified=False,
                              verbose_eval =None)
          # print(cv_results)
          return np.min(cv_results['rmse-mean'])
  # Domain space-- Range of hyperparameters 
  # pds = {'num_leaves': sp_randint(6, 50),
  #           'feature_fraction': (0.1, 0.9),
  #           'bagging_fraction': (0.8, 1),
  #           'max_depth': (17, 25),
  #           'min_split_gain': (0.001, 0.1),
  #           'min_child_weight': (10, 25)
  #           }
  pds = {'num_iterations': 100,
         'learning_rate': tuple(np.linspace(0.001, 0.05, 10)), 
         'early_stopping_round':10,
         'metric':'rmse',
         'num_leaves': (6, 50, 100), 
         'min_child_samples': (100, 300, 400, 500), 
         'min_child_weight': (1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4),
         # 'subsample': sp_uniform(loc=0.2, scale=0.8), 
         # 'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
         'reg_alpha': (0, 1e-1, 1, 2, 5, 7, 10, 50, 100),
         'reg_lambda': (0, 1e-1, 1, 5, 10, 20, 50, 100),
         'feature_fraction': tuple(np.linspace(0.1, 0.9, num=6)),
         'bagging_fraction': tuple(np.linspace(0.8, 1, num=5)),
         'max_depth': tuple(np.linspace(10, 15, num=5)),
         'n_estimators': tuple(np.linspace(2000, 5000, num=4))
         }

  # Surrogate model
  optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)
                                    
  # Optimize
  optimizer.maximize(init_points=init_iter, n_iter=n_iters)
  
  return optimizer

optimizer = bayesion_opt_lgbm(train_data, init_iter=5, n_iters=10, 
                              random_state=77, seed = 101)



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
