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
labels = df_train['country_destination'].values
#df_train = df_train.drop(['country_destination'], axis=1)
id_train = df_train['id']
id_test = df_test['id']
piv_train = df_train.shape[0]

age = pd.read_csv('G:/Geek World/Kaggle/Airbnb/Input/age_buckets_edit.csv')

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

# df_all.dac_tfa_diff[df_all.dac_tfa_diff > 0].describe()
# import matplotlib.pyplot as plt
# import math
# x = df_all.tfa_hour
# plt.hist(x.values)
# plt.title("Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()

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
average_age   = df_all["age"].mean()
std_age      = df_all["age"].std()
count_nan_age = df_all["age"].isnull().sum()

# av = df_all.age.values
# df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

### *********CHANGE THIS FASSSSST*********
rand_1 = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)
df_all["age"][np.isnan(df_all["age"])] = rand_1

#null_df = df_all[pd.isnull(df_all.iloc[:,0:62]).any(axis=1)]
##df_all.iloc[:,0:62].apply(lambda x : x.isnull().sum())
#len(df_all.id)

#df_all.gender.unique()
df_all.first_affiliate_tracked[pd.isnull(df_all.first_affiliate_tracked)] = '-unknown-'
#df_all.gender[df_all.gender == '-unknown-'] = None

#len(df_all.gender[df_all.gender == '-unknown-'])


### AGE CODE
df_all['age_bucket'] = pd.cut(df_all["age"], [15, 20, 25, 30, 35,40,45,50,55,60,65,70,75,80,85,90,95,100,101], labels = ['15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+'], right = True)
#test.age = pd.cut(test["age"], [15, 20, 25, 30, 35,40,45,50,55,60,65,70,75,80,85,90,95,100,101], labels = ['15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80-84','85-89','90-94','95-99','100+'], right = True)


df_all = pd.merge(df_all, age, left_on=['age_bucket', 'gender'], right_on=['age_bucket','gender'], how='left')
#test = pd.merge(test, age, left_on=['age','gender'], right_on=['age_bucket','gender'], how='left')

# ##------------xxxxxxxxxxxx--------------------


df_all['language'][~(df_all['language'].isin(['de','es','fr','it','nl','pt','en']))]= 'other'
#test['language'][~(test['language'].isin(['de','es','fr','it','nl','pt','en']))]= 'other'

countries_edited = pd.read_csv('G:/Geek World/Kaggle/Airbnb/Input/countries_edited.csv')
df_all = pd.merge(df_all, countries_edited, left_on='language', right_on='language', how='left')
#test = pd.merge(test, countries_edited, left_on='language', right_on='language', how='left')



#### SESSIONS CODE
sessions = pd.read_csv('G:/Geek World/Kaggle/Airbnb/Input/sessions.csv')
sessions.info()

total_secs_elapsed = sessions.groupby('user_id', as_index = False).sum()
import matplotlib.pyplot as plt
import scipy.stats
#total_secs_elapsed['secs_elapsed'] = pd.Series(scipy.stats.boxcox(total_secs_elapsed['secs_elapsed'].add(1).fillna(1))[0])
secs_elapsed_by_action = sessions.groupby(['user_id', 'action'], as_index = False).sum()
secs_elapsed_by_action = secs_elapsed_by_action.pivot_table(index='user_id', columns='action', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_action.reset_index(inplace = True)

secs_elapsed_by_action_type = sessions.groupby(['user_id', 'action_type'], as_index = False).sum()
secs_elapsed_by_action_type = secs_elapsed_by_action_type.pivot_table(index='user_id', columns='action_type', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_action_type.reset_index(inplace = True)

secs_elapsed_by_device_type = sessions.groupby(['user_id', 'device_type'], as_index = False).sum()
secs_elapsed_by_device_type = secs_elapsed_by_device_type.pivot_table(index='user_id', columns='device_type', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_device_type.reset_index(inplace = True)

device_brand_map = {'-unknown-': 'other', 'Android App Unknown Phone/Tablet': 'android', 'Android Phone' : 'android', 'Blackberry': 'other', 'Chromebook':'other', 'Linux Desktop':'other', 'Mac Desktop': 'apple', 'Opera Phone':'other','Tablet':'other', 'Windows Desktop': 'windows', 'Windows Phone':'windows', 'iPad Tablet':'apple', 'iPhone':'apple','iPodtouch':'apple'}
device_screen_map = {'-unknown-': 'unknown', 'Android App Unknown Phone/Tablet': 'unknown', 'Android Phone' : 'small', 'Blackberry': 'small', 'Chromebook':'big', 'Linux Desktop':'big', 'Mac Desktop': 'big', 'Opera Phone':'small','Tablet':'medium', 'Windows Desktop': 'big', 'Windows Phone':'small', 'iPad Tablet':'medium', 'iPhone':'small','iPodtouch':'small'}
##num_country_dic = {y:x for x,y in country_num_dic.items()}

sessions['device_brand'] = pd.Series(sessions.device_type).map(device_brand_map)
sessions['device_screen'] = pd.Series(sessions.device_type).map(device_screen_map)

secs_elapsed_by_device_brand = sessions.groupby(['user_id', 'device_brand'], as_index = False).sum()
secs_elapsed_by_device_brand = secs_elapsed_by_device_brand.pivot_table(index='user_id', columns='device_brand', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_device_brand.reset_index(inplace = True)

secs_elapsed_by_device_screen = sessions.groupby(['user_id', 'device_brand'], as_index = False).sum()
secs_elapsed_by_device_screen = secs_elapsed_by_device_screen.pivot_table(index='user_id', columns='device_brand', values='secs_elapsed', fill_value = 0)
secs_elapsed_by_device_screen.reset_index(inplace = True)

sessions_merged = pd.merge(total_secs_elapsed, secs_elapsed_by_device_type, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, secs_elapsed_by_device_brand, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, secs_elapsed_by_device_screen, left_on='user_id', right_on='user_id', how='left')

t = sessions.groupby("user_id",as_index = False).agg({"action" : {'unique_action' : lambda x: x.nunique(), 'total_act' : len} , "action_type" : {'unique_act_types' : lambda x: x.nunique(), 'total_act_types' : len}, "device_type" : {'unique_device_types' : lambda x: x.nunique(), 'total_device_types' : len} })
t_lens = sessions.groupby("user_id",as_index = False).agg({"action" : len, "action_type" : len, "device_type" : len })
t_uniques = sessions.groupby("user_id",as_index = False).agg({"action" : lambda x: x.nunique(), "action_type" : lambda x: x.nunique(), "device_type" : lambda x: x.nunique()})
t_uniques.columns = ['user_id', 'unique_actions', 'unique_act_types', 'unique_device_types']
t_lens.columns = ['user_id', 'total_actions', 'total_act_types', 'total_device_types']
t_all = pd.merge(t_uniques, t_lens, left_on='user_id', right_on = 'user_id', how='outer')


sessions_merged = pd.merge(sessions_merged, secs_elapsed_by_action, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, secs_elapsed_by_action_type, left_on='user_id', right_on='user_id', how='left')
sessions_merged = pd.merge(sessions_merged, t_all, left_on='user_id', right_on='user_id', how='left')

df_all = pd.merge(df_all, sessions_merged, left_on='id', right_on='user_id', how='left')
df_all = df_all.drop(['user_id'], axis=1)
df_all.iloc[:,93:] = df_all.iloc[:,93:].apply(lambda x: pd.Series(scipy.stats.boxcox(x.add(1).fillna(1))[0]), axis = 0)


#----------------xxxxxxxxxxxxx------------------

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

param = {'bst:max_depth':6, 'bst:eta':0.1, 'silent':0, 'objective':'multi:softprob', 'num_class':12, 'eval_metric':'mlogloss', 'nthread':-1}


#scores:  XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=50
#0.8183974444336767
Y_test_sample = test_sample["country_destination"]
Y_test_sample = Y_test_sample.map(country_num_dic)

X_train_sample.isnull().sum()

eval_set  = [(X_train_sample, Y_train_sample), (X_test_sample, Y_test_sample)]
dtrain = xgb.DMatrix(X_train_sample, label=Y_train_sample)
dtest = xgb.DMatrix(X_test_sample, label=Y_test_sample)

evallist  = [(dtest,'eval'), (dtrain,'train')]

xgb_fit = xgb.train(params= param, dtrain = dtrain, evals=evallist, num_boost_round=75)
#xgb = xgb.train(params=param , subsample=0.5, colsample_bytree=0.5, seed=321)
#Y_pred_sample = xgb_fit.predict(X_test_sample)
Y_pred_sample = xgb_fit.predict(dtest)


import operator
xgb_fit.get_fscore()
importance = xgb_fit.get_fscore()
#importance = sorted(importance.items(), key=operator.itemgetter(1))

imp_df = pd.DataFrame.from_dict(importance, orient='index')
imp_df.reset_index(inplace=True)
imp_df.columns = ['variable', 'gini_index']

imp_df.to_csv('G:/Geek World/Kaggle/Airbnb/model_importance_xgb_wuniqe_sessions.csv',index=False)

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


# ----------- Again to create csv of the predictions ------------------
ids = []  #list of ids
for i in range(len(id_test_sample)):
    idx = id_test[i]
    ids += [idx] * 5

sub_sample = pd.Series(ids).to_frame(name = 'id')
sub_sample['country'] = pd.Series(cts)

sub_sample.to_csv('G:/Geek World/Kaggle/Airbnb/sub_sample_12jan.csv',index=False)
##-----------------XXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------------------------------------


##------------------ NOW TRAIN MODEL ON FULL DATA ------------------------

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

submission.to_csv('G:/Geek World/Kaggle/Airbnb/sub_full_12jan.csv',index=False)

