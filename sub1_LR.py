#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 06:40:18 2018

@author: yuyingjie
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
subm = pd.read_csv('sample_submission.csv')

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

preds = np.zeros((test.shape[0], len(col)))
nrow_train=train.shape[0]
loss = []

for i, j in enumerate(col):
    print('===Fit '+j)
    model = LogisticRegression()
    model.fit(df_w2v_tfidf[:nrow_train], train[j])
    preds[:,i] = model.predict_proba(df_w2v_tfidf[nrow_train:])[:,1]
    
    pred_train = model.predict_proba(df_w2v_tfidf[:nrow_train])[:,1]
    print('ROC AUC:', roc_auc_score(train[j], pred_train))
    loss.append(roc_auc_score(train[j], pred_train))
    
print('mean column-wise ROC AUC:', np.mean(loss))
    
    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission.csv', index=False)