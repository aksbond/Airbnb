import numpy as np
import pandas as pd
import time
import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier

np.random.seed(0)

#Loading data
df_train = pd.read_csv('G:/Geek World/Kaggle/Airbnb/Input/train_users_2.csv')
df_test = pd.read_csv('G:/Geek World/Kaggle/Airbnb/Input/test_users.csv')

# df_train = df_train.iloc[0:4000,:]
# df_test = df_test.iloc[0:4000,:]

labels = df_train['country_destination'].values
#df_train = df_train.drop(['country_destination'], axis=1)
id_train = df_train['id']
id_test = df_test['id']
piv_train = df_train.shape[0]

age = pd.read_csv('G:/Geek World/Kaggle/Airbnb/Input/age_buckets_edit2.csv')

len(df_train[(df_train.date_first_booking.isnull()) & (df_train.country_destination == 'NDF')])
len(df_train)


#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

##  Date_account_created
### This has only date attributes - Get weekday, weekend flag, day, month , year, holiday or not, ** Make month categorical **
## Get # of days before next holiday,
df_all.date_account_created.describe()
df_all.date_account_created.head(n=10)

#dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = df_all.date_account_created.apply(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%Y')))
df_all['dac_month'] = df_all.date_account_created.apply(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%m'))-1)
df_all['dac_day'] = df_all.date_account_created.apply(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%d')))
df_all.loc[:,['date_account_created','dac_year', 'dac_month', 'dac_day']].head(n=10)
#df_all = df_all.drop(['date_account_created'], axis=1)
import datetime

#df_all['date_account_created'] = df_all.date_account_created.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
df_all.date_account_created.head(n=10)
df_all['dac_weekday'] = df_all.date_account_created.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%A'))
#df_all['dac_weekday'].unique()
df_all['dac_weekend_flag'] = df_all.dac_weekday.map(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
#df_all['dac_weekend_flag'].unique()
df_all['dac_ext_weekend_flag'] = df_all.dac_weekday.map(lambda x: 1 if x in ['Friday', 'Saturday', 'Sunday'] else 0)
#df_all['dac_ext_weekend_flag'].unique()


#timestamp_first_active - Weekday attributes, Time of the day when active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = df_all.timestamp_first_active.apply(lambda x: int(datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S').strftime('%Y')))
df_all['tfa_month'] = df_all.timestamp_first_active.apply(lambda x: int(datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S').strftime('%m'))-1)
df_all['tfa_day'] = df_all.timestamp_first_active.apply(lambda x: int(datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S').strftime('%d')))
df_all['tfa_hour'] = df_all.timestamp_first_active.apply(lambda x: int(datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S').strftime('%H'))) + df_all.timestamp_first_active.apply(lambda x: int(datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S').strftime('%M'))/60)
#df_all['tfa_hour'].describe()

#df_all['timestamp_first_active'] = df_all.timestamp_first_active.apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S'))
df_all['tfa_weekday'] = df_all.timestamp_first_active.apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S').strftime('%A'))
#df_all['tfa_weekday'].unique()
df_all['tfa_weekend_flag'] = df_all.tfa_weekday.map(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
#df_all['tfa_weekend_flag'].unique()
df_all['tfa_ext_weekend_flag'] = df_all.tfa_weekday.map(lambda x: 1 if x in ['Friday', 'Saturday', 'Sunday'] else 0)
#df_all['tfa_ext_weekend_flag'].unique()


## Difference between TFA and DAC in hours
df_all['dac_tfa_diff'] = df_all.timestamp_first_active.apply(lambda x: time.mktime(datetime.datetime.strptime(str(x), '%Y%m%d%H%M%S').timetuple())) - df_all.date_account_created.apply(lambda x: time.mktime(datetime.datetime.strptime(x, '%Y-%m-%d').timetuple()))
df_all['dac_tfa_diff'] = df_all.dac_tfa_diff.apply(lambda x : x/(3600))
df_all['dac_tfa_diff'].describe()

## Get number of days left to the closest holiday  --- ***Calculate weights for each holiday***
## *****Month number will be from 0 to 11*********
holiday_list = pd.read_csv('G:/Geek World/Kaggle/Airbnb/Input/USholidays.csv')
holiday_list['month'] = holiday_list.hol_date.apply(lambda x: int(datetime.datetime.strptime(x, '%d-%m-%Y').strftime('%m'))-1)
holiday_list['year'] = holiday_list.hol_date.apply(lambda x: int(datetime.datetime.strptime(x, '%d-%m-%Y').strftime('%Y')))
holiday_list.tail(n=1)

user_dac = df_all.loc[:,['id','date_account_created', 'dac_month', 'dac_year']]
#user_dac.head(n=1)
user_dac = pd.merge(user_dac, holiday_list, left_on=['dac_year'], right_on=['year'], how='left')
user_dac['d2h1'] = (user_dac.hol_date.apply(lambda x: time.mktime(datetime.datetime.strptime(str(x), '%d-%m-%Y').timetuple())) - user_dac.date_account_created.apply(lambda x: time.mktime(datetime.datetime.strptime(x, '%Y-%m-%d').timetuple())))/(3600*24)
user_dac['d2h2'] = (user_dac.next_hol_date.apply(lambda x: time.mktime(datetime.datetime.strptime(str(x), '%d-%m-%Y').timetuple())) - user_dac.date_account_created.apply(lambda x: time.mktime(datetime.datetime.strptime(x, '%Y-%m-%d').timetuple())))/(3600*24)
user_dac.d2h2.isnull().sum()
user_dac['days_to_hol'] = user_dac.apply(lambda row: row.d2h2 if row.d2h1 < 0 else row.d2h1, axis = 1)
user_dac.head(n=10)

user_dac = user_dac.loc[:,['id','hol_desc', 'days_to_hol']]
user_dac= user_dac.pivot_table(index='id', columns='hol_desc', values='days_to_hol', fill_value = 365)
user_dac.reset_index(inplace = True)
user_dac.head(n=10)
len(user_dac)
len(df_all)

df_all = pd.merge(df_all, user_dac, left_on='id', right_on='id', how='left')
len(df_all)
df_all.head(n=2)


#df_all = df_all.drop(['timestamp_first_active'], axis=1)


#Age
df_all.loc[df_all.age<15,'age']=15
df_all.loc[df_all.age>100,'age']=100

# get average, std, and number of NaN values in airbnb_df
# average_age   = df_all["age"].mean()
# std_age      = df_all["age"].std()
# count_nan_age = df_all["age"].isnull().sum()

# av = df_all.age.values
# df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

### *********CHANGE THIS FASSSSST*********
#rand_1 = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)
#df_all["age"][np.isnan(df_all["age"])] = rand_1

null_df = df_all[pd.isnull(df_all.iloc[:,0:62]).any(axis=1)]
df_all.iloc[:,0:62].apply(lambda x : x.isnull().sum())
len(df_all.id)

#df_all.gender.unique()
df_all.first_affiliate_tracked[pd.isnull(df_all.first_affiliate_tracked)] = '-unknown-'
df_all.gender[df_all.gender == '-unknown-'] = None

#len(df_all.gender[df_all.gender == '-unknown-'])



df_all['language'][~(df_all['language'].isin(['de','es','fr','it','nl','pt','en']))]= 'other'
#test['language'][~(test['language'].isin(['de','es','fr','it','nl','pt','en']))]= 'other'

countries_edited = pd.read_csv('G:/Geek World/Kaggle/Airbnb/Input/countries_edited.csv')
df_all = pd.merge(df_all, countries_edited, left_on='language', right_on='language', how='left')
#test = pd.merge(test, countries_edited, left_on='language', right_on='language', how='left')



## ------------- Get session attributes ---------------------------------------------
sessions = pd.read_csv('G:/Geek World/Kaggle/Airbnb/Input/sessions.csv')

sessions_mod = sessions.loc[:,['action', 'action_type']]
sessions_mod.action[(sessions_mod.action.isnull()) | (sessions_mod.action == '-unknown-')] = 'Act_unknown'
sessions_mod.action_type[(sessions_mod.action_type.isnull()) | (sessions_mod.action_type == '-unknown-')] = 'Actype_unknown'
#sessions_mod.action_detail[(sessions_mod.action_detail.isnull()) | (sessions_mod.action_detail == '-unknown-')] = 'Actdetail_unknown'

sessions_mod.action.nunique()

# Get distinct number of action types for each action
action_dist_types = pd.DataFrame(sessions_mod.groupby(['action'], as_index = False).agg({"action_type" : lambda x: x.nunique()}))
# action_dist_types = sessions_mod.groupby(['action_type'], as_index = False).agg({"action" : lambda x: x.nunique()})

# Get those actions having more than 1 action type
duplicate_act_types = action_dist_types[action_dist_types.action_type > 1]
duplicate_act_types.columns = ['action', 'num_act_types']
# len(action_dist_types)

# Total counts of action types vs actions
action_types_uniques = pd.DataFrame({'count' : sessions_mod.groupby( ["action", "action_type"] ).size()}).reset_index()
action_types_uniques.tail(n=10)
# action_types_uniques[action_types_uniques.action == 'approve']

# Remove the ones that are in duplicate and action type is unknown
action_types_uniques = action_types_uniques[~(action_types_uniques.action.isin(duplicate_act_types.action) & (action_types_uniques.action_type == 'Actype_unknown' ))]
# len(action_types_uniques)
# action_types_uniques.action.nunique()

## Get maximum of all duplicate actions' action types
## The unknowns will be replaced with the most frequently occurring action type
act_max_ind = action_types_uniques.groupby(["action"])['count'].transform(max) == action_types_uniques['count']
action_types_uniques_max = action_types_uniques[act_max_ind]
action_types_uniques_max.head(n=5)
len(action_types_uniques_max)

# Create map of action to action type
act_dict = dict(zip(action_types_uniques_max.action, action_types_uniques_max.action_type))
# act_dict

# Subset for train and test users
sessions = sessions[sessions.user_id.isin(df_all.id)]

sessions.action[(sessions.action.isnull()) | (sessions.action == '-unknown-')] = 'Act_unknown'
sessions.action_type[(sessions.action_type.isnull()) | (sessions.action_type == '-unknown-')] = 'Actype_unknown'
sessions.action_detail[(sessions.action_detail.isnull()) | (sessions.action_detail == '-unknown-')] = 'Acdetail_unknown'
sessions.action.nunique()
sessions.action_type.nunique()

sessions.action_type = sessions.action.map(act_dict)
action_types_uniques = pd.DataFrame({'count': sessions.groupby(["action", "action_type"]).size()}).reset_index()
len(action_types_uniques) ## If 360 then good to go

# sessions.head(n=10)
import scipy.stats
total_secs_elapsed = sessions.groupby('user_id', as_index = False).sum()

secs_elapsed_by_action = sessions.groupby(['user_id', 'action'], as_index = False).sum()
# secs_elapsed_by_action.head(n=5)
# secs_elapsed_by_action.action.unique()
action_list = pd.read_csv("G:/Geek World/Kaggle/Airbnb/Input/Actions.csv")
# secs_elapsed_by_action = secs_elapsed_by_action[secs_elapsed_by_action.action.isin(action_list.Actions.unique())]
secs_elapsed_by_action.loc[~(secs_elapsed_by_action['action'].isin(action_list.Actions.unique())), 'action'] = 'Other_actions'
secs_elapsed_by_action = secs_elapsed_by_action.pivot_table(index='user_id', columns='action', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_action.reset_index(inplace = True)

secs_elapsed_by_action_type = sessions.groupby(['user_id', 'action_type'], as_index = False).sum()
secs_elapsed_by_action_type.head(n=5)
action_type_list = pd.read_csv("G:/Geek World/Kaggle/Airbnb/Input/Action_types.csv")
secs_elapsed_by_action_type.loc[~(secs_elapsed_by_action_type['action_type'].isin(action_type_list.Action_type.unique())), 'action_type'] = 'Other_action_types'
secs_elapsed_by_action_type = secs_elapsed_by_action_type.pivot_table(index='user_id', columns='action_type', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_action_type.reset_index(inplace = True)


secs_elapsed_by_action_detail = sessions.groupby(['user_id', 'action_detail'], as_index = False).sum()
# secs_elapsed_by_action_detail.head(n=5)
action_detail_list = pd.read_csv("G:/Geek World/Kaggle/Airbnb/Input/Action_details.csv")
secs_elapsed_by_action_detail.loc[~(secs_elapsed_by_action_detail['action_detail'].isin(action_detail_list.Action_details.unique())), 'action_detail'] = 'Other_action_details'
secs_elapsed_by_action_detail = secs_elapsed_by_action_detail.pivot_table(index='user_id', columns='action_detail', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_action_detail.reset_index(inplace = True)

secs_elapsed_by_device_type = sessions.groupby(['user_id', 'device_type'], as_index = False).sum()
secs_elapsed_by_device_type = secs_elapsed_by_device_type.pivot_table(index='user_id', columns='device_type', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_device_type.reset_index(inplace = True)

device_brand_map = {'-unknown-': 'other', 'Android App Unknown Phone/Tablet': 'android', 'Android Phone' : 'android', 'Blackberry': 'other', 'Chromebook':'other', 'Linux Desktop':'other', 'Mac Desktop': 'apple', 'Opera Phone':'other','Tablet':'other', 'Windows Desktop': 'windows', 'Windows Phone':'windows', 'iPad Tablet':'apple', 'iPhone':'apple','iPodtouch':'apple'}
device_screen_map = {'-unknown-': 'Dev_unknown', 'Android App Unknown Phone/Tablet': 'Dev_unknown', 'Android Phone' : 'small', 'Blackberry': 'small', 'Chromebook':'big', 'Linux Desktop':'big', 'Mac Desktop': 'big', 'Opera Phone':'small','Tablet':'medium', 'Windows Desktop': 'big', 'Windows Phone':'small', 'iPad Tablet':'medium', 'iPhone':'small','iPodtouch':'small'}
##num_country_dic = {y:x for x,y in country_num_dic.items()}

sessions['device_brand'] = pd.Series(sessions.device_type).map(device_brand_map)
sessions.device_brand[sessions.device_brand == '-unknown-'] = 'other'
sessions['device_screen'] = pd.Series(sessions.device_type).map(device_screen_map)
sessions.device_screen[sessions.device_screen == '-unknown-'] = 'Dev_unknown'

secs_elapsed_by_device_brand = sessions.groupby(['user_id', 'device_brand'], as_index = False).sum()
secs_elapsed_by_device_brand = secs_elapsed_by_device_brand.pivot_table(index='user_id', columns='device_brand', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_device_brand.reset_index(inplace = True)

secs_elapsed_by_device_screen = sessions.groupby(['user_id', 'device_brand'], as_index = False).sum()
secs_elapsed_by_device_screen = secs_elapsed_by_device_screen.pivot_table(index='user_id', columns='device_brand', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_device_screen.reset_index(inplace = True)



## Frequency and number of unique events
t_lens = sessions.groupby("user_id",as_index = False).agg({"action" : len, "action_type" : len, "action_detail" : len, "device_type" : len })
t_uniques = sessions.groupby("user_id",as_index = False).agg({"action" : lambda x: x.nunique(), "action_type" : lambda x: x.nunique(), "action_detail" : lambda x: x.nunique(), "device_type" : lambda x: x.nunique()})

x = list(['total_' + elt for elt in t_lens.columns[1:]])
x.insert(0,'user_id')
t_lens.columns = x


t_uniques = sessions[(sessions.action.isin(action_list.Actions.unique())) | (sessions.action_type.isin(action_type_list.Action_type.unique())) | (sessions.action_detail.isin(action_detail_list.Action_details.unique()))].groupby("user_id",as_index = False).agg({"action" : lambda x: x.nunique(), "action_type" : lambda x: x.nunique(), "action_detail" : lambda x: x.nunique(), "device_type" : lambda x: x.nunique()})
x = list(['unique_' + elt for elt in t_uniques.columns[1:]])
x.insert(0,'user_id')
t_uniques.columns = x


### Get the shortlisted column frequency
t_imp_lens = sessions[(sessions.action.isin(action_list.Actions.unique())) | (sessions.action_type.isin(action_type_list.Action_type.unique())) | (sessions.action_detail.isin(action_detail_list.Action_details.unique()))].groupby("user_id",as_index = False).agg({"action" : len, "action_type" : len, "action_detail" : len, "device_type" : len })
## Applying appropriate names
x = list(['total_imp_' + elt for elt in t_imp_lens.columns[1:]])
x.insert(0,'user_id')
t_imp_lens.columns = x

t_imp_uniques = sessions[(sessions.action.isin(action_list.Actions.unique())) | (sessions.action_type.isin(action_type_list.Action_type.unique())) | (sessions.action_detail.isin(action_detail_list.Action_details.unique()))].groupby("user_id",as_index = False).agg({"action" : lambda x: x.nunique(), "action_type" : lambda x: x.nunique(), "action_detail" : lambda x: x.nunique(), "device_type" : lambda x: x.nunique()})
x = list(['unique_imp_' + elt for elt in t_imp_uniques.columns[1:]])
x.insert(0,'user_id')
t_imp_uniques.columns = x


t_lens['tot_act_type_ratio'] = t_lens['total_action_type']/t_lens['total_action']
t_lens['tot_act_detail_ratio'] = t_lens['total_action_detail']/t_lens['total_action']

## Unique only becasue action types to action ratio for a user will be almost 1
t_uniques['uniq_act_type_ratio'] = t_uniques['unique_action_type']/t_uniques['unique_action']
t_uniques['uniq_act_detail_ratio'] = t_uniques['unique_action_detail']/t_uniques['unique_action']

t_uniques['uniq_actype_detail_ratio'] = t_uniques['unique_action_type']/t_uniques['unique_action_detail']

t_all = pd.merge(t_uniques, t_lens, left_on='user_id', right_on = 'user_id', how='outer')
t_all = pd.merge(t_all, t_imp_lens, left_on='user_id', right_on = 'user_id', how='outer')
t_all = pd.merge(t_all, t_imp_uniques, left_on='user_id', right_on = 'user_id', how='outer')

t_all['action_imp_total_ratio'] = t_all['total_imp_action']/t_all['total_action']
t_all['action_imp_type_total_ratio'] = t_all['total_imp_action_type']/t_all['total_action_type']
t_all['action_imp_detail_total_ratio'] = t_all['total_imp_action_detail']/t_all['total_action_detail']

t_all['total_action_type_imp_ratio'] = t_all['total_imp_action_type']/t_all['total_imp_action']
t_all['total_action_detail_imp_ratio'] = t_all['total_imp_action_detail']/t_all['total_imp_action']
t_all['total_action_detail_type_imp_ratio'] = t_all['total_imp_action_detail']/t_all['total_imp_action_type']

t_all['action_imp_unique_ratio'] = t_all['unique_imp_action']/t_all['unique_action']
t_all['action_imp_type_unique_ratio'] = t_all['unique_imp_action_type']/t_all['unique_action_type']
t_all['action_imp_detail_unique_ratio'] = t_all['unique_imp_action_detail']/t_all['unique_action_detail']

t_all['unique_action_type_imp_ratio'] = t_all['unique_imp_action_type']/t_all['unique_imp_action']
t_all['unique_action_detail_imp_ratio'] = t_all['unique_imp_action_detail']/t_all['unique_imp_action']
t_all['unique_action_detail_type_imp_ratio'] = t_all['unique_imp_action_detail']/t_all['unique_imp_action_type']


sessions_merged = pd.merge(total_secs_elapsed, secs_elapsed_by_device_type, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, secs_elapsed_by_device_brand, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, secs_elapsed_by_device_screen, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, secs_elapsed_by_action, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, secs_elapsed_by_action_type, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, secs_elapsed_by_action_detail, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, t_all, left_on='user_id', right_on='user_id', how='left')

###--------------xxxxxxxxxxxxxxxxxxxxxxxxxxx-----------------------------xxxxxxxxxxxxxxxxxxxxxxxxxxx-------------------------

df_all = pd.merge(df_all, sessions_merged, left_on='id', right_on='user_id', how='left')
df_all = df_all.drop(['user_id'], axis=1)
df_all.iloc[:,82:] = df_all.iloc[:,82:].apply(lambda x: pd.Series(scipy.stats.boxcox(x.add(1).fillna(1))[0]), axis = 0)


df_all.id[df_all.id.isin(sessions_merged.user_id)]

del(sessions)
del(sessions_mod)
del(secs_elapsed_by_action)
del(secs_elapsed_by_action_type)
del(secs_elapsed_by_device_brand)
del(secs_elapsed_by_action_detail)
del(secs_elapsed_by_device_screen)
del(t_lens)
del(t_uniques)
del(t_all)

mvt_base = df_all.drop(['country_destination', 'date_account_created', 'timestamp_first_active', 'date_first_booking'], axis=1)

ohe_feats = ['dac_weekday', 'tfa_weekday', 'most_probable_country', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    mvt_dummy = pd.get_dummies(mvt_base[f], prefix=f)
    mvt_base = mvt_base.drop([f], axis=1)
    mvt_base = pd.concat((mvt_base, mvt_dummy), axis=1)


mvt_base = mvt_base.drop(['id'], axis = 1)

len(mvt_base[~(mvt_base.gender.isnull()) & ~(mvt_base.age.isnull())])
# Total entries in the base = 275547
## Total entries with complete data = 136444
gender_train_Y = mvt_base.loc[~(mvt_base.gender.isnull()) & ~(mvt_base.age.isnull()),'gender']
gender_num_dic = {'OTHER': 0, 'FEMALE': 1, 'MALE': 2}
num_gender_dic = {y:x for x,y in gender_num_dic.items()}
gender_train_Y = gender_train_Y.map(gender_num_dic)

pd.Series(gender_train_Y).unique()
len(gender_train_Y)

pd.crosstab(gender_train_Y, 1)

age_train_Y = mvt_base.loc[~(mvt_base.gender.isnull()) & ~(mvt_base.age.isnull()),'age']
age_train_Y.unique()
len(age_train_Y)

mvt_train_X = mvt_base[~(mvt_base.gender.isnull()) & ~(mvt_base.age.isnull())]
len(mvt_train_X)


##mvt_train_X.iloc[:,0:23].head(n=2)

mvt_valid_X = mvt_base[~(mvt_base.gender.isnull()) & (mvt_base.age.isnull())]
gender_valid_Y = mvt_base.loc[~(mvt_base.gender.isnull()) & (mvt_base.age.isnull()),'gender']
gender_num_dic = {'OTHER': 0, 'FEMALE': 1, 'MALE': 2}
num_gender_dic = {y:x for x,y in gender_num_dic.items()}
gender_valid_Y = gender_valid_Y.map(gender_num_dic)

mvt_test_X = mvt_base[mvt_base.gender.isnull()]


# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(mvt_train_X.drop(['age', 'gender'], axis=1), gender_train_Y)
# gender_valid_pred = clf.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))

np.random.seed(321)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(mvt_train_X.drop(['age', 'gender'], axis=1), gender_train_Y)
gender_valid_pred = neigh.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))

pd.crosstab(gender_valid_pred,gender_valid_Y)

gender_test_pred = neigh.predict(mvt_test_X.drop(['age', 'gender'], axis=1))

### Create id and predicted gender data frame gender
len(df_all.gender[df_all.gender.isnull()])
len(gender_test_pred)

df_all.gender[df_all.gender.isnull()] = gender_test_pred

######------- Predict the age ---------------
age_train_Y = mvt_base.loc[~(mvt_base.gender.isnull()) & ~(mvt_base.age.isnull()),'age']
age_train_Y.unique()
len(age_train_Y)

mvt_train_X = mvt_base[~(mvt_base.gender.isnull()) & ~(mvt_base.age.isnull())]
len(mvt_train_X)

##mvt_train_X.iloc[:,0:23].head(n=2)

mvt_valid_X = mvt_base[(mvt_base.gender.isnull()) & ~(mvt_base.age.isnull())]
age_valid_Y = mvt_base.loc[(mvt_base.gender.isnull()) & ~(mvt_base.age.isnull()),'age']

mvt_test_X = mvt_base[mvt_base.age.isnull()]
len(mvt_test_X )

import sklearn.ensemble as sken
# import sklearn.pipeline as skpipe
# import sklearn.preprocessing as skpreproc

## Random forest for age prediction
rf_sample_train = sken.RandomForestRegressor(n_estimators=300, n_jobs = -1, max_depth=6,
                                     warm_start = False, oob_score = True, verbose=1, random_state=321)

rf_sample_train.fit(mvt_train_X.drop(['age', 'gender'], axis=1), age_train_Y)
rf_age_valid_pred = rf_sample_train.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))

import statistics

# len(age_valid_pred)
# len(age_valid_Y)

error_df = pd.DataFrame(pd.Series(rf_age_valid_pred), columns=['pred'])
error_df.reset_index( inplace=True)
act_df = pd.DataFrame(age_valid_Y)
act_df.reset_index( inplace=True)
error_df = pd.concat([act_df,error_df], axis=1, ignore_index=True)
error_df.columns = ['ind', 'actual', 'ind2', 'pred']
error_df = error_df.drop(['ind','ind2'], axis = 1)

error_df['sqerr'] = error_df.apply(lambda row:(row.pred - row.actual)*(row.pred - row.actual), axis = 1)
statistics.mean(error_df.sqerr)
#error_df.age.isnull().sum()


error_df[error_df.pred.isnull()]


pd.Series(rf_age_valid_pred).isnull().sum()
statistics.mean(error_df.sqerr)

len(rf_age_valid_pred)
len(age_valid_Y)
pd.crosstab(pd.Series(rf_age_valid_pred).apply(lambda x: round(x)),age_valid_Y)

## XGB for age prediction
from xgboost.sklearn import XGBRegressor

xgb = XGBRegressor(max_depth=6, learning_rate=0.2, n_estimators=100,
                    objective='reg:linear', subsample=0.5, colsample_bytree=0.5, seed=321)

eval_set  = [(mvt_train_X.drop(['age', 'gender'], axis=1), age_train_Y), (mvt_valid_X.drop(['age', 'gender'], axis=1),age_valid_Y)]
xgb.fit(mvt_train_X.drop(['age', 'gender'], axis=1), age_train_Y, eval_set = eval_set, eval_metric= 'rmse',early_stopping_rounds= 10, verbose=1)
xgb_age_valid_pred = xgb.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))


## ADAboost for age prediction
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(n_estimators=50,learning_rate=0.1,loss='linear', random_state=321)
ada.fit(mvt_train_X.drop(['age', 'gender'], axis=1), age_train_Y.values,)
ada_age_valid_pred = ada.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))

len(ada_age_valid_pred)
len(age_valid_Y)

error_df = pd.DataFrame(pd.Series(ada_age_valid_pred), columns=['pred'])
error_df.reset_index( inplace=True)
act_df = pd.DataFrame(age_valid_Y)
act_df.reset_index( inplace=True)
error_df = pd.concat([act_df,error_df], axis=1, ignore_index=True)
error_df.columns = ['ind', 'actual', 'ind2', 'pred']
error_df = error_df.drop(['ind','ind2'], axis = 1)

error_df['sqerr'] = error_df.apply(lambda row:(row.pred - row.actual)*(row.pred - row.actual), axis = 1)
statistics.mean(error_df.sqerr)
## Random forest validation MSE - 277.6592940975284
## KNN 3 MSE - 431.43926788685525
## KNN 5 MSE - 460.45001573953323
## ADA boost - 297.33651953144431
## XGB MSE - 272.29326345322602



# Ensemble of the three models to get the best predictions
rf_age_valid_pred = rf_sample_train.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))
ada_age_valid_pred = ada.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))
xgb_age_valid_pred  = xgb.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))

rf_age_valid_pred = pd.DataFrame(pd.Series(rf_age_valid_pred), columns=['pred'])
rf_age_valid_pred.reset_index(inplace=True)
ada_age_valid_pred = pd.DataFrame(ada_age_valid_pred)
ada_age_valid_pred.reset_index(inplace=True)
xgb_age_valid_pred = pd.DataFrame(xgb_age_valid_pred)
xgb_age_valid_pred.reset_index(inplace=True)
act_df = pd.DataFrame(age_valid_Y)
act_df.reset_index( inplace=True)

error_df = pd.concat([act_df, rf_age_valid_pred,ada_age_valid_pred, xgb_age_valid_pred], axis = 1)
error_df = error_df.drop(['index'], axis = 1)
error_df.columns = ['actual', 'pred_rf','pred_ada', 'pred_xgb']
error_df['avg_pred'] = error_df.apply(lambda row: (0.1 * row.pred_rf + 0.1 * row.pred_ada + 0.8 * row.pred_xgb), axis = 1)
error_df['sqerr'] = error_df.apply(lambda row:(row.avg_pred - row.actual)*(row.avg_pred - row.actual), axis = 1)
statistics.mean(error_df.sqerr)


rf_age_test_pred = rf_sample_train.predict(mvt_test_X.drop(['age', 'gender'], axis=1))
ada_age_test_pred = ada.predict(mvt_test_X.drop(['age', 'gender'], axis=1))
xgb_age_test_pred  = xgb.predict(mvt_test_X.drop(['age', 'gender'], axis=1))

rf_age_test_pred = pd.DataFrame(pd.Series(rf_age_test_pred), columns=['pred'])
rf_age_test_pred.reset_index(inplace=True)
ada_age_test_pred = pd.DataFrame(ada_age_test_pred)
ada_age_test_pred.reset_index(inplace=True)
xgb_age_test_pred = pd.DataFrame(xgb_age_test_pred)
xgb_age_test_pred.reset_index(inplace=True)

error_df = pd.concat([rf_age_test_pred,ada_age_test_pred, xgb_age_test_pred], axis = 1)
error_df = error_df.drop(['index'], axis = 1)
error_df.columns = ['pred_rf','pred_ada', 'pred_xgb']
error_df['avg_pred'] = error_df.apply(lambda row: (0.1 * row.pred_rf + 0.1 * row.pred_ada + 0.8 * row.pred_xgb), axis = 1)
error_df.avg_pred = error_df.avg_pred.apply(lambda x: round(x))




### Create predicted gender data frame gender
len(df_all.age[df_all.age.isnull()])
len(error_df.avg_pred)
df_all.age[df_all.age.isnull()] = error_df.avg_pred.values

##--------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------
##--------------------------------------------------------------------------------------------------------------------------------
#
#
### Now get some session attributes for the users who don't have any

######------- Predict the age ---------------
df_all = pd.merge(df_all.iloc[:,0:81], sessions_merged, left_on='id', right_on='user_id', how='left')
df_all = df_all.drop(['user_id'], axis=1)
#df_all.iloc[:,82:] = df_all.iloc[:,82:].apply(lambda x: pd.Series(scipy.stats.boxcox(x.add(1).fillna(1))[0]), axis = 0)

#df_all.id[df_all.id.isin(sessions_merged.user_id)]


# Predict secs_elapsed
mvt_base = df_all.iloc[:,0:82].drop(['country_destination', 'date_account_created', 'timestamp_first_active', 'date_first_booking'], axis=1)
mvt_base.iloc[:,80:] = mvt_base.iloc[:,80:].apply(lambda x: pd.Series(scipy.stats.boxcox(x.add(1).fillna(1))[0]), axis = 0)

#
# import matplotlib.pyplot as plt
# plt.hist(mvt_base['secs_elapsed'])
# plt.title("Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()



ohe_feats = ['dac_weekday', 'gender', 'tfa_weekday', 'most_probable_country', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    mvt_dummy = pd.get_dummies(mvt_base[f], prefix=f)
    mvt_base = mvt_base.drop([f], axis=1)
    mvt_base = pd.concat((mvt_base, mvt_dummy), axis=1)

mvt_base = mvt_base.fillna(0)

train = mvt_base[mvt_base.id.isin(sessions_merged.user_id)]
test = mvt_base[~(mvt_base.id.isin(sessions_merged.user_id))]

from sklearn.cross_validation import train_test_split
train_sample, test_sample = train_test_split(train, test_size = 0.25)


train_sample_Y = train_sample.loc[:,'secs_elapsed']
test_sample_Y = test_sample.loc[:,'secs_elapsed']

train_sample_X = train_sample.drop(['id','secs_elapsed'], axis = 1)
test_sample_X = test_sample.drop(['id','secs_elapsed'], axis = 1)
test_X = test.drop(['id', 'secs_elapsed'], axis = 1)

import sklearn.ensemble as sken

## Random forest for age prediction
rf_sample_train = sken.RandomForestRegressor(n_estimators=300, n_jobs = -1, max_depth=6,
                                     warm_start = False, oob_score = True, verbose=1, random_state=321)

rf_sample_train.fit(train_sample_X.values, train_sample_Y.values)

rf_secs_valid_pred = rf_sample_train.predict(test_sample_X)

train_sample_Y.describe()
import statistics

error_df = pd.DataFrame(pd.Series(rf_secs_valid_pred), columns=['pred'])
error_df.reset_index( inplace=True)
act_df = pd.DataFrame(test_sample_Y)
act_df.reset_index( inplace=True)
error_df = pd.concat([act_df,error_df], axis=1, ignore_index=True)
error_df.columns = ['ind', 'actual', 'ind2', 'pred']
error_df = error_df.drop(['ind','ind2'], axis = 1)

error_df['sqerr'] = error_df.apply(lambda row:(row.pred - row.actual)*(row.pred - row.actual), axis = 1)
statistics.mean(error_df.sqerr)
#error_df.age.isnull().sum()
## Error - 3441301380912.5498

error_df[error_df.pred.isnull()]


pd.Series(rf_age_valid_pred).isnull().sum()
statistics.mean(error_df.sqerr)

len(rf_age_valid_pred)
len(age_valid_Y)
pd.crosstab(pd.Series(rf_age_valid_pred).apply(lambda x: round(x)),age_valid_Y)

## XGB for age prediction
from xgboost.sklearn import XGBRegressor

xgb = XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=100,
                    objective='reg:linear', subsample=0.8, colsample_bytree=0.5, seed=321)

eval_set  = [(train_sample_X, train_sample_Y), (test_sample_X, test_sample_Y)]
xgb.fit(train_sample_X, train_sample_Y, eval_set = eval_set, eval_metric= 'rmse',early_stopping_rounds= 10, verbose=1)
xgb_secs_valid_pred = xgb.predict(test_sample_X)



# ## ADAboost for secs prediction
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.tree import DecisionTreeRegressor
# ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=100,learning_rate=0.1,loss='linear', random_state=321)
# ada.fit(train_sample_X, train_sample_Y.values)
# ada_secs_valid_pred = ada.predict(test_sample_X)
#
# len(ada_secs_valid_pred)

import math
error_df = pd.DataFrame(pd.Series(xgb_secs_valid_pred), columns=['pred'])
error_df.reset_index( inplace=True)
act_df = pd.DataFrame(test_sample_Y.values)
act_df.reset_index( inplace=True)
error_df = pd.concat([act_df,error_df], axis=1, ignore_index=True)
error_df.columns = ['ind', 'actual', 'ind2', 'pred']
error_df = error_df.drop(['ind','ind2'], axis = 1)

error_df['sqerr'] = error_df.apply(lambda row:(row.pred - row.actual)*(row.pred - row.actual), axis = 1)
math.sqrt(statistics.mean(error_df.sqerr))
## Random forest validation MSE - 277.6592940975284
## KNN 3 MSE - 431.43926788685525
## KNN 5 MSE - 460.45001573953323
## ADA boost - 297.33651953144431
## XGB MSE - 272.29326345322602



# Ensemble of the three models to get the best predictions
#rf_age_valid_pred = rf_sample_train.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))
#ada_age_valid_pred = ada.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))
#xgb_secs_valid_pred  = xgb.predict(mvt_valid_X.drop(['age', 'gender'], axis=1))

# rf_age_valid_pred = pd.DataFrame(pd.Series(rf_age_valid_pred), columns=['pred'])
# rf_age_valid_pred.reset_index(inplace=True)
# ada_age_valid_pred = pd.DataFrame(ada_age_valid_pred)
# ada_age_valid_pred.reset_index(inplace=True)
# xgb_age_valid_pred = pd.DataFrame(xgb_age_valid_pred)
# xgb_age_valid_pred.reset_index(inplace=True)
# act_df = pd.DataFrame(age_valid_Y)
# act_df.reset_index( inplace=True)
#
# error_df = pd.concat([act_df, rf_age_valid_pred,ada_age_valid_pred, xgb_age_valid_pred], axis = 1)
# error_df = error_df.drop(['index'], axis = 1)
# error_df.columns = ['actual', 'pred_rf','pred_ada', 'pred_xgb']
# error_df['avg_pred'] = error_df.apply(lambda row: (0.1 * row.pred_rf + 0.1 * row.pred_ada + 0.8 * row.pred_xgb), axis = 1)
# error_df['sqerr'] = error_df.apply(lambda row:(row.avg_pred - row.actual)*(row.avg_pred - row.actual), axis = 1)
# statistics.mean(error_df.sqerr)
#
#
# rf_age_test_pred = rf_sample_train.predict(mvt_test_X.drop(['age', 'gender'], axis=1))
# ada_age_test_pred = ada.predict(mvt_test_X.drop(['age', 'gender'], axis=1))
# xgb_age_test_pred  = xgb.predict(mvt_test_X.drop(['age', 'gender'], axis=1))
#
# rf_age_test_pred = pd.DataFrame(pd.Series(rf_age_test_pred), columns=['pred'])
# rf_age_test_pred.reset_index(inplace=True)
# ada_age_test_pred = pd.DataFrame(ada_age_test_pred)
# ada_age_test_pred.reset_index(inplace=True)
# xgb_age_test_pred = pd.DataFrame(xgb_age_test_pred)
# xgb_age_test_pred.reset_index(inplace=True)
#
# error_df = pd.concat([rf_age_test_pred,ada_age_test_pred, xgb_age_test_pred], axis = 1)
# error_df = error_df.drop(['index'], axis = 1)
# error_df.columns = ['pred_rf','pred_ada', 'pred_xgb']
# error_df['avg_pred'] = error_df.apply(lambda row: (0.1 * row.pred_rf + 0.1 * row.pred_ada + 0.8 * row.pred_xgb), axis = 1)
# error_df.avg_pred = error_df.avg_pred.apply(lambda x: round(x))
#
#

xgb_age_test_pred  = xgb.predict(mvt_test_X.drop(['age', 'gender'], axis=1))

### Create predicted gender data frame gender
len(df_all.age[df_all.age.isnull()])
len(error_df.avg_pred)
df_all.age[df_all.age.isnull()] = error_df.avg_pred.values










### ---xxxxxxxx END OF MISSING VALUE TREATMENT xxxxxxxxxxxxxxxx---------------------------



### Get the age buckets to map with other data
df_all['age_bucket'] = pd.cut(df_all["age"], [15, 20, 25, 30, 35,40,45,50,55,60,65,70,75,80,85,90,95,100,101], labels = ['15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+'], right = True)
#test.age = pd.cut(test["age"], [15, 20, 25, 30, 35,40,45,50,55,60,65,70,75,80,85,90,95,100,101], labels = ['15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+'], right = True)

df_all = pd.merge(df_all, age, left_on=['age_bucket', 'gender', 'language'], right_on=['age_bucket','gender', 'language'], how='left')
#test = pd.merge(test, age, left_on=['age','gender'], right_on=['age_bucket','gender'], how='left')

df_all.age.isnull().sum()


#Removing id and date_first_booking
df_all = df_all.drop(['date_first_booking', 'date_account_created', 'timestamp_first_active', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)


#One-hot-encoding features
ohe_feats = ['dac_weekday', 'tfa_weekday', 'age_bucket', 'most_probable_country', 'gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)


df_all.columns.to_series().groupby(df_all.dtypes).groups


### --------- START PREDICTION OF COUNTRY DESTINATION
train = df_all[:piv_train]
train['country_destination'] = df_train['country_destination']
test = df_all[piv_train:df_all.shape[0]]

country_num_dic = {'NDF': 0, 'US': 1, 'other': 2, 'FR': 3, 'IT': 4, 'GB': 5, 'ES': 6, 'CA': 7, 'DE': 8, 'NL': 9, 'AU': 10, 'PT': 11}
num_country_dic = {y:x for x,y in country_num_dic.items()}


## ------------- STRATIFIED SAMPLING ---------------------
from sklearn.cross_validation import train_test_split
train_sample, test_sample = train_test_split(train, test_size = 0.25, stratify = train.country_destination)

Y_train = df_train['country_destination']
Y_train = Y_train.map(country_num_dic)

Y_train_sample = train_sample["country_destination"]
Y_train_sample = Y_train_sample.map(country_num_dic)

Y_test_sample = test_sample["country_destination"]
Y_test_sample = Y_test_sample.map(country_num_dic)
#Y_test = test["country_destination"]

X_train_sample = train_sample.drop(["country_destination", "id"],axis=1)
X_test_sample  = test_sample.drop(["country_destination", "id"],axis=1)
X_train = train.drop(["country_destination", "id"],axis=1)
X_test = test.drop(['country_destination', "id"],axis=1)

import xgboost as xgb
#------------- TRAIN MODEL ON 75% DATA --------------------------

T_train_sample_xgb = xgb.DMatrix(X_train_sample, Y_train_sample)
X_test_sample_xgb = xgb.DMatrix(X_test_sample)


xgb = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=200,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
#scores:  XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=50
#0.8183974444336767 121 rounds
Y_test_sample = test_sample["country_destination"]
Y_test_sample = Y_test_sample.map(country_num_dic)

X_train_sample.isnull().sum()

eval_set  = [(X_train_sample, Y_train_sample), (X_test_sample, Y_test_sample)]
xgb.fit(X_train_sample, Y_train_sample, eval_set = eval_set, eval_metric = 'mlogloss', early_stopping_rounds= 10)
Y_pred_sample = xgb.predict_proba(X_test_sample)


y_le_train_sample = (train_sample['country_destination'].map(country_num_dic)).values
y_le_test_sample = (test_sample['country_destination'].map(country_num_dic)).values
y_le_train = (train['country_destination'].map(country_num_dic)).values

id_train = train['id'].values
id_train_sample = train_sample['id'].values
id_test_sample = test_sample['id'].values
id_test = test['id'].values


#------------- TRAIN SAMPLE PREDICTION --------------------------
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test_sample)):
    ids.append([])
    ids[i] = id_test_sample[i]
    cts += pd.Series(np.argsort(Y_pred_sample[i])[::-1]).map(num_country_dic).values[:5].tolist()

Y_test_sample = Y_test_sample.map(num_country_dic)
data = np.array(cts)
shape = (len(id_test_sample), 5)
test_sample_preds = pd.DataFrame(data.reshape(shape), columns = ['c1','c2','c3','c4','c5'])
print('\n\n scores: \n', np.mean(score_predictions(test_sample_preds, Y_test_sample)))
Y_test_sample = Y_test_sample.map(num_country_dic)

### Data treated
##  NDCG on the sample --  0.8282540863654848

# ----------- Again to create csv of the predictions ------------------
ids = []  #list of ids
for i in range(len(id_test_sample)):
    idx = id_test[i]
    ids += [idx] * 5

sub_sample = pd.Series(ids).to_frame(name = 'id')
sub_sample['country'] = pd.Series(cts)

sub_sample.to_csv('G:/Geek World/Kaggle/Airbnb/sub_sample_13jan_mvt.csv',index=False)
##-----------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------------------------------------


##------------------ NOW TRAIN MODEL ON FULL DATA ------------------------
import xgboost as xgb
T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb = xgb.DMatrix(X_test)



# params = {"objective": "multi:softprob", "num_class": 12, 'bst:eta':0.1, 'nthread':4}
# params['eval_metric'] = 'ndcg'

#bst_cv = xgb.cv(params, T_train_xgb, nfold = 5, num_boost_round = 2, fold = train_sample)

#gbm = xgb.train(params, T_train_xgb, num_boost_round = 20, verbose_eval = True)
#Y_pred = gbm.predict(X_test_xgb)
xgb = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=143,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)

#eval_set  = [(X_train, Y_train), (X_test, Y_test)]
xgb.fit(X_train, Y_train, eval_metric = 'mlogloss')

Y_pred = xgb.predict_proba(X_test)


ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += pd.Series(np.argsort(Y_pred[i])[::-1]).map(num_country_dic).values[:5].tolist()

submission = pd.Series(ids).to_frame(name = 'id')
submission['country'] = pd.Series(cts)

submission.to_csv('G:/Geek World/Kaggle/Airbnb/sub_full_13jan_mvt143.csv',index=False)


submission.head(n=10)