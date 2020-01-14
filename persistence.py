"""
    Saving relevant things.
"""
import csv
import json
import os

import numpy as np
import torch

import pandas as pd


def save_everything(args, dicts, metrics_hist_all, model, optimizer, model_dir, params, criterion, metrics_codes=None, metrics_inst=None, hadm_ids=None, evaluate=False, test_only=False):
    """
        Save metrics, model, params all in model_dir
    """
    save_metrics(dicts, metrics_hist_all, model_dir, metrics_codes=metrics_codes, metrics_inst=metrics_inst, hadm_ids=hadm_ids, test_only=test_only)
    params['model_dir'] = model_dir
    
    if not test_only:
        save_params_dict(params)
    
    if evaluate or test_only:
        print("saved metrics to directory %s\n" % (model_dir))
        return
        
    sd = model.cpu().state_dict()
    torch.save(sd, os.path.join(model_dir, "model_last_epoch.pth"))
    
    sd_opt = optimizer.state_dict()
    sd_opt['epoch'] = metrics_hist_all[0]['epochs']
    torch.save(sd_opt, os.path.join(model_dir, "optim_last_epoch.pth"))
    
    #save the model with the best criterion metric
    if not np.all(np.isnan(metrics_hist_all[1][criterion])):
        if np.nanargmax(metrics_hist_all[1][criterion]) == len(metrics_hist_all[1][criterion]) - 1:
            torch.save(sd, os.path.join(model_dir, "model_best_{}.pth".format(criterion)))
            torch.save(sd_opt, os.path.join(model_dir, "optim_best_{}.pth".format(criterion)))
    if args.gpu:
        model.cuda()
    print("saved metrics, params, model, optmizer state to directory %s\n" % (model_dir))



def save_metrics(dicts, metrics_hist, model_dir, metrics_codes=None, metrics_inst=None, hadm_ids=None, test_only=False):
    filename = "metrics.json" if not test_only else "metrics_test_only.json"
    with open(os.path.join(model_dir, filename), 'w') as metrics_file:
        #concatenate train, dev and test metrics into one dict
        metrics_hist_train, metrics_hist_dev, metrics_hist_test = metrics_hist
        data = {}
        data.update({"%s_train" % (name):val for (name,val) in metrics_hist_train.items()})
        data.update({"%s_dev" % (name):val for (name,val) in metrics_hist_dev.items()})
        if metrics_hist_test is not None:
            data.update({"%s_test" % (name):val for (name,val) in metrics_hist_test.items()})
        json.dump(data, metrics_file, indent=1)
    if metrics_codes is not None and metrics_inst is not None:
        with open(model_dir + '/metrics_per_code.csv', 'w') as f1, open(model_dir + '/metrics_per_doc.csv', 'w') as f2:
            f1_writer = csv.writer(f1, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            f1_writer.writerow(['code', 'desc', 'valid', 'billable', 'rank', 'train examples', 'test examples', 'accuracy', 'precision', 'recall', 'f1'])
            for row in zip(dicts['ind2c'].values(), dicts['desc_plain'], dicts['valid'].values(), dicts['billable'].values(), dicts['rank'].values(), dicts['train_examples'].values(), dicts['test_examples'].values(), *metrics_codes):
                f1_writer.writerow(row)
            f2_writer = csv.writer(f2, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            f2_writer.writerow(['hadm_id', 'accuracy', 'precision', 'recall', 'f1'])
            for row in zip(hadm_ids, *metrics_inst):
                f2_writer.writerow(row)

def save_params_dict(params):
    with open(os.path.join(params["model_dir"], "params.json"), 'w') as params_file:
        json.dump(params, params_file, indent=1)

def get_codes():
    return ['427.31', '584.9', '428.0', '401.9', '518.81',  '787.02', '785.4', '038.12', '501']

#top_10_codes = ['401.9', '38.93', '428.0', '427.31', '414.01', '96.04', '96.6', '584.9', '250.00' ,'96.71']

def write_docs_top_10(model_dir, fold, df, c2ind, desc_plain):

    if not os.path.exists(os.path.join(model_dir, 'qualitative')):
        os.makedirs(os.path.join(model_dir, 'qualitative'))
    docs_dir = 'qualitative'
    
    for i, code in enumerate(get_codes()):
        ind = c2ind[code]
        
        docs_tp = df[df.apply(lambda x: bool(x['target'][ind] == 1) & bool(x['prediction'][ind] == 1), axis=1)]
        #docs_tp.attention = docs_tp['attention'].apply(lambda a : a[ind])
        n_samples = 10 if len(docs_tp)>=10 else len(docs_tp)
        if n_samples > 0:
            docs_tp = docs_tp.sample(n=n_samples)[['id', 'text', 'attention']].reset_index()
            docs_tp['attention'] = docs_tp['attention'].apply(lambda a : int(np.argmax(a[i])))
            out_path = os.path.join(model_dir, docs_dir, code + '_tp.csv')
            docs_tp[['text', 'attention']].apply(lambda row : ' '.join(row['text'][(row['attention']-7 if row['attention']-7 >= 0 else 0):row['attention']+8]), axis=1).to_csv(out_path, sep=' ', index=False, header=[desc_plain[ind]])
            #out_path = model_dir + '/docs_' + code + '_tp_' + fold + '.json'
            #with open(out_path, 'w') as f:
            #    json.dump(json.loads(docs_tp.sample(n=n_samples)[['id', 'text', 'attention']].to_json(orient='records')), f, indent=1)
        
        docs_fn = df[df.apply(lambda x: bool(x['target'][ind] == 1) & bool(x['prediction'][ind] == 0), axis=1)]
        n_samples = 10 if len(docs_fn)>=10 else len(docs_fn)
        if n_samples > 0:
            docs_fn = docs_fn.sample(n=n_samples)[['id', 'text', 'attention']].reset_index()
            docs_fn['attention'] = docs_fn['attention'].apply(lambda a : int(np.argmax(a[i])))
            out_path = os.path.join(model_dir, docs_dir, code + '_fn.csv')
            docs_fn[['text', 'attention']].apply(lambda row : ' '.join(row['text'][(row['attention']-7 if row['attention']-7 >= 0 else 0):row['attention']+8]), axis=1).to_csv(out_path, sep=' ', index=False, header=[desc_plain[ind]])

def write_docs(model_dir, fold, df, lower, upper):
    assert upper >= lower
    
    docs = df[(df['f1']>=lower) & (df['f1']<=upper)]
    n_samples = 5 if len(docs)>=5 else len(docs)
    docs = docs.sample(n=n_samples)
    
    if len(docs) < 1:
        return
            
    i = 0
    for _, doc in docs.iterrows():
        i += 1
        tp = np.nonzero((doc['target'] == 1) & (doc['prediction'] == 1))[0]
        #tn = [int(c) for c in (doc['target'] == 0) & (doc['prediction'] == 0) if c]
        fp = [int(c) for c in (doc['target'] == 0) & (doc['prediction'] == 1) if c]
        fn = [int(c) for c in (doc['target'] == 1) & (doc['prediction'] == 0) if c]
            
        doc_tp = pd.DataFrame()
        doc_tp['id'] = [n for n in range(len(tp))]
        doc_tp['text'] = [doc['text'] for n in range(len(tp))]
        doc_tp['attention'] = [a for a in doc['attention'][tp]]
        doc_tp['label'] = [t for t in tp]
        doc_tp['prediction'] = [p for p in tp]
        doc_tp['posterior'] = [0 for p in tp] 

        out_path = os.path.join(model_dir, 'doc_f1_' + str(int(lower*100)) + '_' + str(int(upper*100)) + '_' + str(i) + '_tp_' + fold + '.json')
        with open(out_path, 'w') as f:
            json.dump(json.loads(doc_tp.to_json(orient='records')), f, indent=1)
            
            
        doc_fp = pd.DataFrame()
        doc_fp['id'] = [n for n in range(len(fp))]
        doc_fp['text'] = [doc['text'] for n in range(len(fp))]
        doc_fp['attention'] = [a for a in doc['attention'][fp,:]]
        doc_fp['label'] = [t for t in fp]
        doc_fp['prediction'] = [p for p in fp]
        doc_fp['posterior'] = [0 for p in fp]
        
        out_path = model_dir + '/doc_f1_' + str(int(lower*100)) + '_' + str(int(upper*100)) + '_' + str(i) + '_fp_' + fold + '.json'
        with open(out_path, 'w') as f:
            json.dump(json.loads(doc_fp.to_json(orient='records')), f, indent=1)
            
        doc_fn = pd.DataFrame()
        doc_fn['id'] = [n for n in range(len(fn))]
        doc_fn['text'] = [doc['text'] for n in range(len(fn))]
        doc_fn['attention'] = [a for a in doc['attention'][fn]]
        doc_fn['label'] = [t for t in fn]
        doc_fn['prediction'] = [p for p in fn]
        doc_fn['posterior'] = [0 for p in fn]
                
        out_path = model_dir + '/doc_f1_' + str(int(lower*100)) + '_' + str(int(upper*100)) + '_' + str(i) + '_fn_' + fold + '.json'
        with open(out_path, 'w') as f:
            json.dump(json.loads(doc_fn.to_json(orient='records')), f, indent=1)

def write_preds(hids, docs, attns, ys, yhats, yhats_raw, metrics, model_dir, fold, ind2c, c2ind, desc_plain):
        
    preds_file = os.path.join(model_dir, 'preds_{}.psv'.format(fold))
    with open(preds_file, 'w') as f:
        w = csv.writer(f, delimiter='|')
        for yhat, hid in zip(yhats, hids):
            codes = sorted([ind2c[ind] for ind in np.nonzero(yhat)[0]])
            if len(codes) == 0:
                w.writerow([hid, ''])
            else:
                w.writerow([hid] + list(codes))
                
    preds_file_target = os.path.join(model_dir, 'preds_{}_target.psv'.format(fold))
    with open(preds_file_target, 'w') as f:
        w = csv.writer(f, delimiter='|')
        for y, hid in zip(ys, hids):
            codes = sorted([ind2c[ind] for ind in np.nonzero(y)[0]])
            if len(codes) == 0:
                w.writerow([hid, ''])
            else:
                w.writerow([hid] + list(codes))
                

    metrics_df = pd.DataFrame(data={'id':hids, 'accuracy':metrics[0], 'precision':metrics[1], 'recall':metrics[2], 'f1':metrics[3]})

    docs_df = pd.DataFrame(data={'id':hids, 'text':docs, 'attention':attns, 'target':list(ys), 'prediction':list(yhats)})
    
    docs_df = docs_df.merge(metrics_df, on='id', how='inner')
    
    write_docs_top_10(model_dir, fold, docs_df, c2ind, desc_plain)
    
