#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# parameters

model_name = "RF"
output_file = f'model_{model_name}.bin'

# data preparation

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.DEATH_EVENT.values
y_test = df_test.DEATH_EVENT.values

del df_full_train['DEATH_EVENT']
del df_test['DEATH_EVENT']



# training the final model

print('training the final model')

rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=5 ,random_state=1)
rf.fit(df_full_train,y_full_train)
y_pred = rf.predict_proba(df_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump(rf, f_out)

print(f'the model is saved to {output_file}')