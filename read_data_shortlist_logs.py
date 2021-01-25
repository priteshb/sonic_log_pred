# -*- coding: utf-8 -*-
"""
SPE GCS Competition
"""

import lasio, os, collections
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('max_columns', None)

file_list = []
file_list += [file for file in os.listdir(os.curdir) if file.endswith(".las")]


missing_value = [-9999.25]
mnemonics = []
response_var = 'DTSM'
log_mapping = pd.read_excel("Logs_Mapping.xlsx", sheet_name="Distinct mnemonics")


#Check well relative locations:
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


#Identify the best log in a dict
def find_best_logs(df, log_dict, response_var):
    for k, v in log_dict.items():
        if len(v) > 1:
            corr = df[v].corrwith(df[response_var])
            best_log = corr.idxmax()
            log_dict[k] = [best_log]
    
    return log_dict
        

#Identify which logs have multiple mappings in a dict
def identify_duplicate_log(log_dict):
    ret_dict = collections.defaultdict(list)
    for k, v in log_dict.items():
        ret_dict[v[0]].append(k)
        
    return ret_dict


#Rename and shortlist logs
def log_renaming_shortlisting(df, log_map, response_var):
    # df_new = pd.DataFrame()
    col_maps = {}
    try:
        for col in df.columns:
            new_col = log_map.loc[log_map['LOG'] == col, 'CATEGORY'].values
            if pd.notna(new_col):
                col_maps[col] = new_col
     
        col_maps = identify_duplicate_log(col_maps)
        col_maps = find_best_logs(df, col_maps, response_var)
        col_maps = {v[0]: k for k, v in col_maps.items()}
        df = df[df.columns.intersection(list(col_maps.keys()))]
        df = df.rename(columns = col_maps)
        
    except Exception as e :
            print(e)
    
    return df


def remove_high_missing_columns(file, df, thresh):
    # print(file, df.columns[df.isnull().mean() > thresh])
    return df[df.columns[df.isnull().mean() < thresh]]
    


mnemonics_df = pd.DataFrame(columns=['FILE', 'LOG', 'UNIT', 'DESC', 'COUNT', 
                                     'MEAN', 'STD', 'MIN', '25%', '50%', '75%', 
                                     'MAX', 'MISSINGNESS'])

#Read files and get log stats in dataframe
for file in file_list:
    file = '00a60e5cc262_TGS.las'
    inputlas = lasio.read(file) #Read file
    print(file)
    df = inputlas.df() #Convert data to dataframe
    df = df.rename_axis("DEPT").reset_index() #Create depth axis and reset index
    df = df.replace(missing_value, '') #Convert missing value validationt o null
    df = df.dropna(subset=['DTSM'])
    des = pd.DataFrame(df.describe()) #Get data stats
    
    for curves in inputlas.curves:
        # if 'SFL' in curves.mnemonic:
        # print(file)
        # if curves.mnemonic not in mnemonics:
            # print(curves.mnemonic)
        curv_desc = [file, curves.mnemonic, curves.unit, curves.descr]
        curv_stats = list(des.loc[:, curves.mnemonic].values)
        missingness = 100*df[curves.mnemonic].isnull().sum()/len(df)
        curv_desc.extend(curv_stats)
        curv_desc.extend([missingness])
        temp_df = pd.DataFrame([curv_desc], 
                               columns=['FILE', 'LOG', 'UNIT', 'DESC', 'COUNT', 
                                        'MEAN', 'STD', 'MIN', '25%', '50%', 
                                        '75%', 'MAX', 'MISSINGNESS'])
        temp_df = temp_df[temp_df['COUNT'] > 0]
        mnemonics_df = mnemonics_df.append(temp_df)  
        
    df = df.dropna(axis=1, how='all')
    df = log_renaming_shortlisting(df, log_mapping, response_var)
    df = remove_high_missing_columns(file, df, 0.05)
    
    
m = 100*df.isnull().sum()/len(df)

    
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
