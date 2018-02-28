#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:07:44 2018

@author: yuyingjie
"""
import pandas as pd
import numpy as np

tfidf = pd.read_csv('submission_tfidf.csv')
cnn=pd.read_csv('sub_w2v_cnn.csv')
mlp = pd.read_csv('sub_w2v_mlp.csv')
keras1 = pd.read_csv('submission_keras1.csv')
keras2 = pd.read_csv('submission_keras2.csv')

b1 = tfidf.copy()
col = tfidf.columns

col = col.tolist()
col.remove('id')

for i in col:
    b1[i] = (2 * tfidf[i] + cnn[i] + mlp[i] + keras1[i] + keras2[i]) / 6
    
b1.to_csv('sub_blending.csv', index = False)

'''
# Technique by the1owl... Thanks 
sub1 = b1[:]
sub2 = ble[:]
cols = [c for c in ble.columns if c not in ['id','comment_text']]
sub2.columns = [x+'_' if x not in ['id'] else x for x in sub2.columns]
blend = pd.merge(sub1, sub2, how='left', on='id')
for c in cols:
    blend[c] = np.sqrt(blend[c] * blend[c+'_'])
    blend[c] = blend[c].clip(0+1e12, 1-1e12)
blend = blend[sub1.columns]
blend.to_csv('hight_of_blending.csv', index = False)
'''