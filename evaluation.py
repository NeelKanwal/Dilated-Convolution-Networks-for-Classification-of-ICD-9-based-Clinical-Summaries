"""
    This file contains evaluation methods that take in a set of predicted labels 
        and a set of ground truth labels and calculate precision, recall, jaccard index, f1, and metrics @k
"""
from collections import defaultdict
import csv
import json
import numpy as np
import os
import sys

from sklearn.metrics import roc_curve, precision_recall_curve, auc
from tqdm import tqdm

import datasets

def all_metrics(yhat, y, k=8, yhat_raw=None, level=None):
    """
        Inputs:
            yhat: binary predictions matrix 
            y: binary ground truth matrix
            k: for @k metrics
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    
    names = ["jac", "prec", "rec", "f1"]

    #macro
    macro, macro_codes = all_macro(yhat, y)

    #micro
    ymic = y.ravel()
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_macro" : macro[i] for i in range(len(macro))}
    metrics.update({names[i] + "_micro" : micro[i] for i in range(len(micro))})

    #AUC and @k
    #allow k to be passed as int or list
    if type(k) != list:
        k = [k]
    for k_i in k:
        rec_at_k = recall_at_k(yhat_raw, y, k_i)
        metrics['rec_at_%d' % k_i] = rec_at_k
        prec_at_k = precision_at_k(yhat_raw, y, k_i)
        metrics['prec_at_%d' % k_i] = prec_at_k
        metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)

    roc_auc, pr_auc = auc_metrics(yhat_raw, y, ymic)
    metrics.update(roc_auc)
    metrics.update(pr_auc)
    
    if level is not None:        
        metrics = {"{}_{}".format(k, level) : v for k,v in metrics.items()}

    metrics_inst_list = inst_jaccard_list(yhat, y), inst_precision_list(yhat, y), inst_recall_list(yhat, y), inst_f1_list(yhat, y)
    
    return metrics, macro_codes, metrics_inst_list

def all_macro(yhat, y):
    #with warnings.catch_warnings():
    #    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    jac, jac_codes = macro_jaccard(yhat, y)
    prec, prec_codes = macro_precision(yhat, y)
    rec, rec_codes = macro_recall(yhat, y)
    f1, f1_codes = macro_f1(yhat, y)
    macro = jac, prec, rec, f1
    macro_codes = jac_codes, prec_codes, rec_codes, f1_codes
    return macro, macro_codes

def all_micro(yhatmic, ymic):
    return micro_jaccard(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

#########################################################################
#MACRO METRICS: calculate metric for each label and average across labels
#########################################################################

def macro_jaccard(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num), num

def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num), num

def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num), num

def macro_f1(yhat, y):
    prec, prec_codes = macro_precision(yhat, y)
    rec, rec_codes = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    f1_codes = 2*(prec_codes*rec_codes)/(prec_codes+rec_codes)
    f1_codes[np.isnan(f1_codes)] = 0.
    return f1, f1_codes

###################
# INSTANCE-AVERAGED
###################

def inst_jaccard_list(yhat, y):
    num = intersect_size(yhat, y, 1) / union_size(yhat, y, 1)
    num[np.isnan(num)] = 0.
    return num

def inst_precision(yhat, y):
    num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)
    
def inst_precision_list(yhat, y):
    num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return num

def inst_recall(yhat, y):
    num = intersect_size(yhat, y, 1) / y.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return np.mean(num)
    
def inst_recall_list(yhat, y):
    num = intersect_size(yhat, y, 1) / y.sum(axis=1)
    #correct for divide-by-zeros
    num[np.isnan(num)] = 0.
    return num

def inst_f1(yhat, y):
    prec = inst_precision(yhat, y)
    rec = inst_recall(yhat, y)
    f1 = 2*(prec*rec)/(prec+rec)
    return f1

def inst_f1_list(yhat, y):
    prec = inst_precision_list(yhat, y)
    rec = inst_recall_list(yhat, y)
    f1 = 2*(prec*rec)/(prec+rec)
    f1[np.isnan(f1)] = 0.
    return f1

##############
# AT-K
##############

def recall_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

def precision_at_k(yhat_raw, y, k):
    #num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]

    #get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))

    return np.mean(vals)

##########################################################################
#MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_jaccard(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (union_size(yhatmic, ymic, 0) + 1e-10)

def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (yhatmic.sum(axis=0) + 1e-10)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (ymic.sum(axis=0) + 1e-10)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def auc_metrics(yhat_raw, y, ymic):
    if yhat_raw.shape[0] <= 1:
        return

    roc_aucs = []
    roc_auc = {}
    for i in range(y.shape[1]):
        fpr, tpr, _ = roc_curve(y[:,i], yhat_raw[:,i])
        roc_aucs.append(auc(fpr, tpr))
    roc_auc['roc_auc_macro'] = np.nanmean(roc_aucs)

    yhatmic = yhat_raw.ravel()
    fpr, tpr, _ = roc_curve(ymic, yhatmic) 
    roc_auc["roc_auc_micro"] = auc(fpr, tpr)
    
    pr_aucs = []
    pr_auc = {}
    for i in range(y.shape[1]):
        prec, rec, _ = precision_recall_curve(y[:,i], yhat_raw[:,i])
        pr_aucs.append(auc(rec, prec))
    pr_auc['pr_auc_macro'] = np.nanmean(pr_aucs)
    
    prec, rec, _ = precision_recall_curve(ymic, yhatmic)
    pr_auc['pr_auc_micro'] = auc(rec, prec)
    
    return roc_auc, pr_auc

########################
# METRICS BY CODE TYPE
########################

def results_by_type(Y, mdir):
    d2ind = {}
    p2ind = {}

    #get predictions for diagnoses and procedures
    diag_preds = defaultdict(lambda: set([]))
    proc_preds = defaultdict(lambda: set([]))
    preds = defaultdict(lambda: set())
    with open('%s/preds_test.psv' % mdir, 'r') as f:
        r = csv.reader(f, delimiter='|')
        for row in r:
            if len(row) > 1:
                for code in row[1:]:
                    preds[row[0]].add(code)
                    if code != '':
                        try:
                            pos = code.index('.')
                            if pos == 3 or (code[0] == 'E' and pos == 4):
                                if code not in d2ind:
                                    d2ind[code] = len(d2ind)
                                diag_preds[row[0]].add(code)
                            elif pos == 2:
                                if code not in p2ind:
                                    p2ind[code] = len(p2ind)
                                proc_preds[row[0]].add(code)
                        except:
                            if len(code) == 3 or (code[0] == 'E' and len(code) == 4):
                                if code not in d2ind:
                                    d2ind[code] = len(d2ind)
                                diag_preds[row[0]].add(code)
    #get ground truth for diagnoses and procedures
    diag_golds = defaultdict(lambda: set([]))
    proc_golds = defaultdict(lambda: set([]))
    golds = defaultdict(lambda: set())
    test_file = '%s/test_%s.csv' % (MIMIC_3_DIR, str(Y))
    with open(test_file, 'r') as f:
        r = csv.reader(f)
        #header
        next(r)
        for row in r:
            codes = set([c for c in row[3].split(';')])
            for code in codes:
                golds[row[1]].add(code)
                try:
                    pos = code.index('.')
                    if pos == 3:
                        if code not in d2ind:
                            d2ind[code] = len(d2ind)
                        diag_golds[row[1]].add(code)
                    elif pos == 2:
                        if code not in p2ind:
                            p2ind[code] = len(p2ind)
                        proc_golds[row[1]].add(code)
                except:
                    if len(code) == 3 or (code[0] == 'E' and len(code) == 4):
                        if code not in d2ind:
                            d2ind[code] = len(d2ind)
                        diag_golds[row[1]].add(code)

    hadm_ids = sorted(set(diag_golds.keys()).intersection(set(diag_preds.keys())))

    ind2d = {i:d for d,i in d2ind.items()}
    ind2p = {i:p for p,i in p2ind.items()}
    type_dicts = (ind2d, ind2p)
    return diag_preds, diag_golds, proc_preds, proc_golds, golds, preds, hadm_ids, type_dicts


def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def print_metrics(metrics, level=None):
    print()
    if level is not None:
        print('metrics ' + level + ' level')
    else: level = ''
        
    print('{:>15s}{:>12s}{:>9s}{:>12s}{:>10s}{:>9s}'.format('[MACRO] ', 'precision', 'recall', 'f-measure', 'ROC AUC', 'PR AUC'))
    print('{:>15.4f}{:>12.4f}{:>9.4f}{:>12.4f}{:>10.4f}{:>9.4f}'.format(*[metrics['{}_{}_{}'.format(k,'macro',level)] for k in ["prec", "rec", "f1", "roc_auc", "pr_auc"]]))
        
    print('{:>15s}{:>12s}{:>9s}{:>12s}{:>10s}{:>9s}'.format('[MICRO] ', 'precision', 'recall', 'f-measure', 'ROC AUC', 'PR AUC'))
    print('{:>15.4f}{:>12.4f}{:>9.4f}{:>12.4f}{:>10.4f}{:>9.4f}'.format(*[metrics['{}_{}_{}'.format(k,'micro',level)] for k in ["prec", "rec", "f1", "roc_auc", "pr_auc"]]))
    
    for metric, val in metrics.items():
        if metric.find("rec_at") != -1:
            print("%s: %.4f" % (metric.replace('_'+level,''), val))
    print()
