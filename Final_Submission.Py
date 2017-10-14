# import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
random.seed(2474)

import xgboost as xgb

# Import training and testing data
train_df = pd.read_csv('data/train/train_submissions.csv')
test_df = pd.read_csv('data/test_submissions_NeDLEvX.csv')

# Import user and problem data
user_df = pd.read_csv('data/train/user_data.csv')
problem_df = pd.read_csv('data/train/problem_data.csv')
print('done reading input..')

# Merge user data with train
train_df = train_df.merge(user_df, on = 'user_id', how = 'left')

# Merge problem data with train
train_df = train_df.merge(problem_df, on = 'problem_id', how = 'left')

# Fill NaN with zero
train_df.fillna(0, inplace = True)
print('train shape is {}'.format(train_df.shape))

# Merge user data with test
test_df = test_df.merge(user_df, on = 'user_id', how = 'left')

# Merge problem data with test
test_df = test_df.merge(problem_df, on = 'problem_id', how = 'left')

# Fill NaN with zero
test_df.fillna(0, inplace = True)
print('test shape is {}'.format(test_df.shape))

# Label encoding of categorical variables
for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values) + list(test_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))

# Feature Engineering
# user features
tmp1 = train_df[['user_id','attempts_range']].groupby(['user_id']).mean().reset_index()
tmp1.columns = ['user_id', 'user_mean_ar']
tmp2 = train_df[['user_id','attempts_range']].groupby(['user_id']).median().reset_index()
tmp2.columns = ['user_id', 'user_median_ar']
tmp3 = train_df[['user_id','attempts_range']].groupby(['user_id']).std().reset_index()
tmp3.columns = ['user_id', 'user_sd_ar']
train_df = train_df.merge(tmp1, on = 'user_id', how = 'left')
train_df = train_df.merge(tmp2, on = 'user_id', how = 'left')
train_df = train_df.merge(tmp3, on = 'user_id', how = 'left')

test_df = test_df.merge(tmp1, on = 'user_id', how = 'left')
test_df = test_df.merge(tmp2, on = 'user_id', how = 'left')
test_df = test_df.merge(tmp3, on = 'user_id', how = 'left')

# problem features
tmp1 = train_df[['problem_id','attempts_range']].groupby(['problem_id']).mean().reset_index()
tmp1.columns = ['problem_id', 'problem_mean_ar']
tmp2 = train_df[['problem_id','attempts_range']].groupby(['problem_id']).median().reset_index()
tmp2.columns = ['problem_id', 'problem_median_ar']
tmp3 = train_df[['problem_id','attempts_range']].groupby(['problem_id']).std().reset_index()
tmp3.columns = ['problem_id', 'problem_sd_ar']
train_df = train_df.merge(tmp1, on = 'problem_id', how = 'left')
train_df = train_df.merge(tmp2, on = 'problem_id', how = 'left')
train_df = train_df.merge(tmp3, on = 'problem_id', how = 'left')

test_df = test_df.merge(tmp1, on = 'problem_id', how = 'left')
test_df = test_df.merge(tmp2, on = 'problem_id', how = 'left')
test_df = test_df.merge(tmp3, on = 'problem_id', how = 'left')

# rank feautures
tmp1 = train_df[['rank','attempts_range']].groupby(['rank']).mean().reset_index()
tmp1.columns = ['rank', 'rank_mean_ar']
tmp2 = train_df[['rank','attempts_range']].groupby(['rank']).median().reset_index()
tmp2.columns = ['rank', 'rank_median_ar']
tmp3 = train_df[['rank','attempts_range']].groupby(['rank']).std().reset_index()
tmp3.columns = ['rank', 'rank_sd_ar']
train_df = train_df.merge(tmp1, on = 'rank', how = 'left')
train_df = train_df.merge(tmp2, on = 'rank', how = 'left')
train_df = train_df.merge(tmp3, on = 'rank', how = 'left')

test_df = test_df.merge(tmp1, on = 'rank', how = 'left')
test_df = test_df.merge(tmp2, on = 'rank', how = 'left')
test_df = test_df.merge(tmp3, on = 'rank', how = 'left')

# contribution features
tmp1 = train_df[['contribution','attempts_range']].groupby(['contribution']).mean().reset_index()
tmp1.columns = ['contribution', 'contrib_mean_ar']
tmp2 = train_df[['contribution','attempts_range']].groupby(['contribution']).median().reset_index()
tmp2.columns = ['contribution', 'contrib_median_ar']
tmp3 = train_df[['contribution','attempts_range']].groupby(['contribution']).std().reset_index()
tmp3.columns = ['contribution', 'contrib_sd_ar']
train_df = train_df.merge(tmp1, on = 'contribution', how = 'left')
train_df = train_df.merge(tmp2, on = 'contribution', how = 'left')
train_df = train_df.merge(tmp3, on = 'contribution', how = 'left')

test_df = test_df.merge(tmp1, on = 'contribution', how = 'left')
test_df = test_df.merge(tmp2, on = 'contribution', how = 'left')
test_df = test_df.merge(tmp3, on = 'contribution', how = 'left')

# problem_rank features
tmp1 = train_df[['problem_id','rank','attempts_range']].groupby(['problem_id','rank']).mean().reset_index()
tmp1.columns = ['problem_id','rank', 'prob_rank_mean_ar']
tmp2 = train_df[['problem_id','rank','attempts_range']].groupby(['problem_id','rank']).median().reset_index()
tmp2.columns = ['problem_id','rank', 'prob_rank_median_ar']
tmp3 = train_df[['problem_id','rank','attempts_range']].groupby(['problem_id','rank']).std().reset_index()
tmp3.columns = ['problem_id','rank', 'prob_rank_sd_ar']
train_df = train_df.merge(tmp1, on = ['problem_id','rank'], how = 'left')
train_df = train_df.merge(tmp2, on = ['problem_id','rank'], how = 'left')
train_df = train_df.merge(tmp3, on = ['problem_id','rank'], how = 'left')

test_df = test_df.merge(tmp1, on = ['problem_id','rank'], how = 'left')
test_df = test_df.merge(tmp2, on = ['problem_id','rank'], how = 'left')
test_df = test_df.merge(tmp3, on = ['problem_id','rank'], how = 'left')

# Labels for training data (dependent variable)
y_train = train_df['attempts_range'].values

# Remove Labels, user_id, problem_id from training features
train = train_df.drop(['attempts_range', 'user_id','problem_id'], axis = 1)

# Remove ID, user_id, problem_id from testing features
test = test_df.drop(['ID','user_id','problem_id'], axis = 1)

# Convert to xgb format
d_train = xgb.DMatrix(train, label=y_train)
d_test = xgb.DMatrix(test)

# Custom evaluation function for f1 score
def f1_score(pred, dtrain):
    labels = dtrain.get_label()
    tp = np.sum(pred.round() == labels)
    fp = len(pred) - tp
    fn = len(labels) - tp
    out = 0
    if tp > 0:
        precision=float(tp)/(tp+fp)
        recall=float(tp)/(tp+fn)
        out = 2*((precision*recall)/(precision+recall))
    else:
        out = 0    
    return 'f1-score', out

# xgb parameters
params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 1,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1
}

# Train model for 40 boosting rounds
model = xgb.train(params, d_train, 40, feval = f1_score, maximize = True)

# Predict on test set
test_pred = model.predict(d_test)

# Save to csv file
out = pd.DataFrame()
out['ID'] = test_df['ID']
out['attempts_range'] = test_pred.round()
out.to_csv('submissions/test_predictions.csv', index = False)
