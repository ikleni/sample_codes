import pandas as pd
import os
import matplotlib.pyplot as plt
import gzip
import csv
import numpy as np
from tqdm import tqdm
import math
from matplotlib import dates as mpl_dates
import matplotlib.ticker as ticker
import matplotlib.cbook as cbook
from datetime import datetime, timedelta
#import seaborn as sns
import pickle
#import statsmodels.api as sm
from scipy import stats, integrate
import pickle
import sys
#import cv2

is_escalator = list(set(['2F#2号扶梯(1Fto2F)进', '3F#1号扶梯(4Fto3F)进', '2F#1号扶梯(1Fto2F)进', '1F#2号扶梯(2Fto1F)进',
               '2F#3号扶梯(3Fto2F)进', '3F#2号扶梯(2Fto3F)进', '3F#3号扶梯(2Fto3F)进', '3F#1号扶梯(2Fto3F)进',
               '3F#3号扶梯(4Fto3F)进', '4F#5号扶梯(3Fto4F)进', '3F#5号扶梯(4Fto3F)进', '2F#1号扶梯(3Fto2F)进',
               '4F#3号扶梯(3Fto4F)进', '4F#2号扶梯(3Fto4F)进', '1F#3号扶梯(2Fto1F)进',  '2F#5号扶梯(3Fto2F)进',
               '3F#5号扶梯(2Fto3F)进', '2F#5号扶梯(3Fto2F)进', '2F#5号扶梯(1Fto2F)进','2F#3号扶梯(1Fto2F)进',
               '4F#1号扶梯(3Fto4F)进', '3F#2号扶梯(4Fto3F)进' , '1F#1号扶梯(2Fto1F)进',  '1F#5号扶梯(2Fto1F)进',
               '2F#2号扶梯(1Fto2F)进','2F#2号扶梯(1Fto2F)进', '3F#3号扶梯(2Fto3F)进', '4F#3号扶梯(3Fto4F)进', '3F#3号扶梯(4Fto3F)进',
 '2F#2号扶梯(3Fto2F)进', '1F#2号扶梯(2Fto1F)进', '2F#3号扶梯(1Fto2F)进', '1F#5号扶梯(2Fto1F)进', '1F#1号扶梯(2Fto1F)进', '3F#2号扶梯(4Fto3F)进',
 '2F#5号扶梯(1Fto2F)进', '4F#1号扶梯(3Fto4F)进', '2F#1号扶梯(3Fto2F)进', '2F#1号扶梯(1Fto2F)进', '2F#5号扶梯(3Fto2F)进', '2F#3号扶梯(3Fto2F)进',
 '3F#5号扶梯(2Fto3F)进', '3F#1号扶梯(2Fto3F)进', '1F#3号扶梯(2Fto1F)进', '3F#2号扶梯(2Fto3F)进', '4F#2号扶梯(3Fto4F)进', '3F#1号扶梯(4Fto3F)进',
 '4F#5号扶梯(3Fto4F)进', '3F#5号扶梯(4Fto3F)进']))


def drop_ids_with_unstable_age_gender(df):

    print('Dropping users who have changing gender and age')

    #create unique personid                   
    df['person_id_new']=df['person_id'].apply(str)+'_'+df['pt_date'].apply(str) 
    # option 2 is do the same thing but also add gender and age

    tmp_df = df[['person_id_new', 'age', 'gender']]

    grouped = tmp_df.groupby(['person_id_new'], as_index = False).nunique()
    grouped['keep'] = (grouped['gender']*grouped['age'] == 1).astype(int)

    df = df.merge(grouped[['person_id_new','keep']], how = 'inner', on = 'person_id_new')
    df = df[df['keep'] == 1]
    df = df.drop(columns = ['keep'])

    N = (tmp_df.shape[0] - df.shape[0])/tmp_df.shape[0] 
    print(f'Removed {N} data points !!!')

    return df

def check_if_entrytime_is_correct_add_new_userid(df):

    print('Correcting datetime and adding daily user id')

    #create unique personid                   
    df['person_id_new']=df['person_id'].apply(str)+'_'+df['pt_date'].apply(str)

    #order columns                  
    df = df[['person_id', 'age', 'gender',  'pt_date', 'person_id_new', 'start_time', 'end_time', 'event_type','duration', 'zone_id', 'zone_name', 'area_type', 'store_id', 'store_name', 'floor_name']]
    #drop duplicates
    df_new = df.drop_duplicates()
    #drop if store_id == 0
    df_new = df_new[(df_new['store_id'] != '0')]


    ########### . Temporal Features & Duration
    df_new["start_time"] = pd.to_datetime(df_new["start_time"])
    df_new["end_time"] = pd.to_datetime(df_new["end_time"])

    df_new["start_hour"] = df_new["start_time"].dt.hour
    df_new["end_hour"] = df_new["end_time"].dt.hour
    df_new["start_min"] = df_new["start_time"].dt.minute
    df_new["end_min"] = df_new["end_time"].dt.minute

    #Monday == 0 … Sunday == 6
    df_new["dayofwk"] = df_new["start_time"].dt.weekday
    df_new["month"] = df_new["start_time"].dt.month

    #duration in the raw data is rounded. (e.g., 0 duration even if 1 sec.): recalculate duration column. 
    df_new['duration'] = (df_new['end_time'] - df_new['start_time']).dt.total_seconds()/60

    # pass by a shop
    df_new['impression'] = np.where(((df_new['event_type']== 3) | (df_new['event_type'] ==12)) & (df_new['store_id'].notnull()), 1, 0)

    # actually enter something
    df_new['visit'] = np.where((df_new['event_type']== 12) & (df_new['store_id'].notnull()) & (df_new['duration'] > 0) , 1, 0)

    #######: Opening time filter: keep entry - exit between 10am - 10pm. 

    ## get mall entry time:
    entry_mall = df_new.groupby(['person_id_new']).agg(entry_time=('start_hour',np.min), exit_time=('end_hour',np.max)).reset_index()
    entry_mall['entry_check'] = np.where((entry_mall['entry_time'] >= 10) & (entry_mall['entry_time']  < 22 ) & (entry_mall['exit_time'] <= 22) , 1, 0)
    # entry_mall['entry_check_2'] = np.where( (entry_mall['exit_time'] - entry_mall['entry_time'] >= 10), 0, 1) # a bit too harsh
    entry_mall = entry_mall[entry_mall.entry_check == 1 ]
    # entry_mall = entry_mall[entry_mall.entry_check_2 == 1 ] 

    # redundant
    # entry_store = df_new[(df_new['visit'] == 1)]
    # entry_store = entry_store.groupby(['person_id_new']).agg(visit_first_entry=('start_hour',np.min), visit_last_exit=('end_hour',np.max)).reset_index()
    # entry_store['visit_check'] = np.where((entry_store['visit_first_entry'] >= 10) 
    #                                       & (entry_store['visit_first_entry']  < 22 ) &  
    #                                       (entry_store['visit_last_exit'] < 22) , 1, 0)
    # entry_store = entry_store[entry_store['visit_check'] == 1]

    # keep 
    df_new_filtered = df_new.merge(entry_mall, how='inner', on='person_id_new', validate = 'many_to_one')
    df_new_filtered = df_new_filtered.drop(columns = ['entry_check'])
#     df_new_filtered = df_new_filtered.drop(columns = ['entry_check_2'])


    N = (df.shape[0] - df_new_filtered.shape[0])/df.shape[0] 
    print(f'Removed {N} data points !!!')

    return df_new_filtered


#if consecutive rows happen within 30 min (e.g., bathroom break)
def dup_store_record_cleanse(df):

    print('Combining disconnected rows with the same stay location')

    N_0 = df.shape[0]
    #df = df.sort_values(["person_id_new", "start_time"])
    df = df[['person_id_new', 'store_id', "start_time", "end_time", "entry_time", "exit_time"]]
    df['time_gap'] = 0
    df = df.reset_index(drop=True)
    max_row = len(df)
    base = 0
    for i, row in df.iterrows():
        #stop if reaches last row. note that we need to subtract 1 since i starts from 0
        if i == max_row-1:
            break
        #identify duplicated rows
        elif (df.loc[base, 'person_id_new'] == df.loc[(base+1), 'person_id_new']) &  (df.loc[base, 'store_id'] == df.loc[(base+1), 'store_id']) & ((df.loc[(base+1), 'start_time']- df.loc[base, 'end_time']) < timedelta(minutes=30)):
            df.loc[base, 'time_gap'] = (df.loc[(base+1), 'start_time']- df.loc[base, 'end_time']).total_seconds()/60
            df.loc[base, 'end_time'] = df.loc[base+1, 'end_time']
            df.drop(base+1, inplace = True)
            df = df.reset_index(drop=True)
            continue
        else:
            #set the comparison to next row. 
            base = base + 1
    df['duration'] = ((df['end_time'] - df['start_time']).dt.total_seconds()/60) - df['time_gap']
    df["start_min"] = df["start_time"].dt.minute
    df["end_min"] = df["end_time"].dt.minute

    N_1 = df.shape[0]
    N = (N_0 - N_1)/N_0

    print(f'Removed {N} data points !!!')

    return df


#if consecutive rows happen within 30 min (e.g., bathroom break)
def dup_store_record_cleanse_vec(df, delta = 30):

    print('Combining disconnected rows with the same stay location')

    N_0 = df.shape[0]

    df = df.sort_values(["person_id_new", "start_time", 'end_time'])

    df['same_zone'] = df.groupby(['person_id_new'], as_index = False)['zone_id'].shift(1) == df[['zone_id']]
    df['same_store']  = df.groupby(['person_id_new'], as_index = False)['store_id'].shift(1) == df[['store_id']]

    # same zone, not nan, time difference between the events is less then 30 minutes
    df['valid_for_merge_type_zone'] = ((df['same_zone'] == 1)&( ~df['zone_id'].isna()))&(((df['start_time']- df['end_time'].shift(1)) < timedelta(minutes=delta)))
    df['valid_for_merge_type_shop'] = ((df['same_store'] == 1)&(~df['store_id'].isna()))&(((df['start_time']- df['end_time'].shift(1)) < timedelta(minutes=delta)))
    df['valid_for_merge'] = df['valid_for_merge_type_zone']|df['valid_for_merge_type_shop'] 

    # detect the final repeating zone (next location is new)
    df['next'] = df.groupby(['person_id_new'], as_index = False)['valid_for_merge'].shift(-1)
    df['next'] = df['next'].fillna(value=False)

    # keep if next location is new or if previous location is different
    df = df[(~df['valid_for_merge']) | (~df['next'])]

    # set end time to the exit time last consecutive
    df['end_time_shifted'] = df.groupby(['person_id_new'], as_index = False)['end_time'].shift(-1)
    df.loc[(~df['valid_for_merge'])*(df['next']), 'end_time'] =  df.loc[(~df['valid_for_merge'])*(df['next']), 'end_time_shifted']


    df=df[~df['valid_for_merge']]
    new_cols = ['end_time_shifted', 'next', 'valid_for_merge', 'valid_for_merge_type_zone',
               'valid_for_merge_type_shop', 'same_zone','same_store']
    df = df.drop(columns = new_cols)


    N_1 = df.shape[0]
    N = (N_0 - N_1)/N_0

    print(f'Removed {N} data points !!!')

    return df

def add_english_names(df):

    print('Adding English names')

    # get english names
    mall_gate_zone = pd.read_csv(f"/Users/IvanK/Malls/english/xagx/xagx_mall_gate_zone.csv")
    store_categories = pd.read_csv(f"/Users/IvanK/Malls/english/xagx/xagx_store_master_english.csv", encoding = 'gbk')

    mall_gate_zone['is_gate'] = 1
    useful_cols = ['?..store_id', 'first_cat', 'second_cat','third_cat', 'area_sqm',
                     'first_cat_english', 'second_cat_english', 'third_cat_english']

    # add english names to the dataframe
    df = df.merge(mall_gate_zone[['zone_id','is_gate']], how = 'left', on = 'zone_id')
    df['is_gate'] = df['is_gate'].fillna(value = 0)
    df = df.merge(store_categories[useful_cols], how = 'left', left_on = 'store_id', right_on = '?..store_id')
    
    # mark escalator zones
    df['is_escalator'] = 0
    
    for z in is_escalator:
        df.loc[df.zone_name == z, 'is_escalator'] = 1
    

    return df
####################################
####################################
####################################
####################################
########## Feature Gener ###########
####################################
####################################
####################################
####################################


def add_user_level_features(df):
    
    print('Adding store & zone visit aggregate features')
    
    df = df.sort_values(by = ['person_id_new','start_time'])
    df = df.reset_index(drop = True)

    # fix impression columns
    df.loc[df.visit == 1,'impression'] = 0

    # get total time inside
    df['delta_mins'] = ( df['start_time'] - df.groupby(['person_id_new'], as_index = False)['end_time'].shift(1)['end_time'] ).dt.total_seconds()/60
    df['delta_mins_too_big'] = (df['delta_mins'] > 20).astype(int)
    df['delta_mins'] = df['delta_mins']*(1-df['delta_mins_too_big'])

    sums = df.groupby(['person_id_new'], as_index = False)['duration'].sum()
    sums['delta_mins'] =  df.groupby(['person_id_new'], as_index = False)['delta_mins'].sum()['delta_mins']
    sums['total_time_inside'] = sums['delta_mins'] + sums['duration']

    # get total visits/impressions
    visits = df.groupby(['person_id_new'], as_index = False).agg(total_visits=('visit',np.sum), total_impressions=('impression',np.sum))

    # add total visits & time inside
    df = df.merge(sums[['total_time_inside', 'person_id_new']], how = 'left' , on = 'person_id_new')
    df = df.merge(visits, how = 'left', on = 'person_id_new')

    # first visit variables
    # FIXME: some visits are split into 2 by a very brief event 

    # adding first interaction with this store
    first_visit_store = df[df['visit'] == 1].groupby(['person_id_new','store_id'], as_index = False)[['start_time', 'duration']].first()
    first_visit_store['is_first_visit'] = 1
    first_visit_store = first_visit_store.rename(columns = {'start_time' : 'first_visit_this_store', 'duration': 'first_store_visit_duration'})

    first_impression_store = df[(df['impression'] == 1)].groupby(['person_id_new','store_id'], as_index = False)['start_time'].first()
    first_impression_store['is_first_impression'] = 1
    first_impression_store = first_impression_store.rename(columns = {'start_time' : 'first_impression_this_store'})

    df = df.merge(first_visit_store, how = 'left', on = ['person_id_new','store_id'])
    df = df.merge(first_impression_store, how = 'left', on = ['person_id_new','store_id'])

    # mark stores that have been visited/seen before today
    df.loc[df['start_time'] != df['first_visit_this_store'], 'is_first_visit'] = 0
    df.loc[df['start_time'] != df['first_impression_this_store'], 'is_first_impression'] = 0
    
    df['revisit'] = (1-df['is_first_visit'])*df['visit']

    # could be that the visit was before first impression
    df['revisit_after_impression'] = (1-df['is_first_impression'])*df['visit']*((df['start_time'] - df['first_impression_this_store'] > timedelta(seconds=0)).astype(int))

    # adding first visit of the day
    # FIXME maybe add first BIG visit instead
    first_visit_of_day = df[df['visit'] == 1].groupby(['person_id_new'], as_index = False)[['start_time', 'store_id', 'duration','first_cat_english', 'second_cat_english', 'third_cat_english' ]].first()
    first_visit_of_day['is_first_visit_of_day'] = 1
    first_visit_of_day = first_visit_of_day.rename(columns = {'start_time' : 'first_visit_today_time','duration': 'first_visit_today_duration'})

    df = df.drop(columns = ['entry_time'])
    entry_point = df.groupby(['person_id_new'], as_index = False)[['start_time', 'zone_id']].first()
    entry_point['entry'] = 1
    entry_point = entry_point.rename(columns = {'start_time' : 'entry_time'})

    first_impression_of_day = df[df['impression'] == 1].groupby(['person_id_new'], as_index = False)[['start_time', 'store_id','first_cat_english', 'second_cat_english', 'third_cat_english' ]].first()
    first_impression_of_day['is_first_impression_of_day'] = 1
    first_impression_of_day = first_impression_of_day.rename(columns = {'start_time' : 'first_impression_today_time'})

    df = df.merge(first_visit_of_day, how = 'left', on = ['person_id_new'] , suffixes = (None,'_first_vis'))
    df = df.merge(first_impression_of_day, how = 'left', on = ['person_id_new'] , suffixes = (None,'_first_impr'))
    df = df.merge(entry_point, how = 'left', on = ['person_id_new'] , suffixes = (None,'_entry'))

    df.loc[df['start_time'] != df['first_visit_today_time'], 'is_first_visit_of_day'] = 0
    df.loc[df['start_time'] != df['first_impression_today_time'], 'is_first_impression_of_day'] = 0
    df.loc[df['start_time'] != df['entry_time'], 'entry'] = 0
    
    df['revisit_time'] = df['revisit']*df['duration']
    df['visit_duration'] = df['visit']*df['duration']

    # useless columns
    df = df.drop(columns = [ 'start_hour', 'end_hour', 'start_min', 'end_min', 'first_cat',
                            'second_cat', 'third_cat'])

    del first_visit_of_day
    del first_visit_store
    del first_impression_store
    del entry_point
    
    return df


####################################
####################################
####################################
####################################
########## Data Cleaning ###########
####################################
####################################
####################################
####################################

print('Processing Data')

# run 
data_list = sys.argv[1:]
print(data_list)
for i in range(len(data_list)):
    
    #read data
    df = pd.read_csv(f'/Users/IvanK/Malls/data/{data_list[i]}')
#     df = pd.read_csv(f'/Users/IvanK/Malls/data/{data_list[i]}', nrows = 1000000)

    print(df.shape)
    df = drop_ids_with_unstable_age_gender(df)
    df = check_if_entrytime_is_correct_add_new_userid(df)
    df = dup_store_record_cleanse_vec(df)
    df = add_english_names(df)
    print(df.shape)

    # adding features
    df = add_user_level_features(df)
    print(df.shape)

    # saving

    df.to_csv(f'/Users/IvanK/Malls/data_with_features/{data_list[i]}', index = False)


