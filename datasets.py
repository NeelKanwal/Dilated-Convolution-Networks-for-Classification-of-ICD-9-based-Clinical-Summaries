"""
    Data loaders
"""
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from collections import defaultdict
import pandas as pd
import csv
import math
import numpy as np
import sys
import os
import jsonlines

import string
import re
from collections import Counter

def load_lookups(args, hier=False):
    """
        Inputs:
            args: Input arguments
        Outputs:
            vocab lookups, ICD code lookups, description lookup, description one-hot vector lookup
    """

    #get code and description lookups
    
    codes, codes_train_examples, codes_test_examples, codes_rank = load_full_codes(args.data_path, hier=hier)
  
    desc_plain, codes_billable = load_code_descriptions(args.data_dir, args.exclude_non_billable)

    if not args.include_invalid:
        codes = codes.intersection(set(desc_plain.keys()))
        
    codes_valid = {c : True for c in codes}
    
    for c in codes.difference(set(desc_plain.keys())):
        codes_billable[c] = False
        codes_valid[c] = False
    
    codes_coarse = set([str(c).split('.')[0] for c in codes if c != ''])
    if hier:
        codes = codes - codes_coarse
        
    ind2c = {i:str(c) for i,c in enumerate(sorted(codes))}
    ind2c_coarse = {i:str(c) for i,c in enumerate(sorted(codes_coarse))}
    
    desc_plain_sorted = [' '.join(desc_plain[code]) for _, code in sorted(ind2c.items())]
    codes_rank_sorted = {code : codes_rank[code] for _, code in sorted(ind2c.items())}
    codes_train_examples_sorted = {code : codes_train_examples.get(code, 0) for _, code in sorted(ind2c.items())}
    codes_test_examples_sorted = {code : codes_test_examples.get(code, 0) for _, code in sorted(ind2c.items())}
    codes_billable_sorted = {code : codes_billable[code] for _, code in sorted(ind2c.items())}
    codes_valid_sorted = {code : codes_valid[code] for _, code in sorted(ind2c.items())}
    
    #get vocab lookups
    codes_desc = [desc_plain[code] for _, code in sorted(ind2c.items())] if args.embed_desc else None
    ind2w, w2ind = load_vocab_dict(args, args.vocab, codes_desc=codes_desc)
    
    #desc = [np.array([int(w2ind.get(word, len(w2ind)+1)) for word in desc_plain[code]], dtype=np.int32) for _, code in sorted(ind2c.items())]
    #m = max([len(text) for text in desc])
    #desc = torch.LongTensor(np.array([np.pad(text, (0, m-len(text)), 'constant') for text in desc]))
    
    desc = [torch.LongTensor([int(w2ind.get(word, len(w2ind)+1)) for word in desc_plain[code]]) for _, code in sorted(ind2c.items())]
    
    desc = torch.nn.utils.rnn.pad_sequence(desc, batch_first=True, padding_value=0)
            
    c2ind = {str(c):i for i,c in ind2c.items()}
    c2ind_coarse = {str(c):i for i,c in ind2c_coarse.items()}
    fine2coarse = np.zeros(len(ind2c))
    for idx, code in ind2c.items():
        idx_coarse = c2ind_coarse[code.split('.')[0]]
        fine2coarse[idx] = idx_coarse

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind, 'ind2c_coarse': ind2c_coarse, 'c2ind_coarse': c2ind_coarse,
             'fine2coarse' : fine2coarse, 'desc': desc, 'desc_plain' : desc_plain_sorted, 'train_examples' : codes_train_examples_sorted, 'test_examples' : codes_test_examples_sorted, 
             'rank' : codes_rank_sorted, 'billable' : codes_billable_sorted, 'valid' : codes_valid_sorted}

    return dicts

def load_vocab_dict(args, vocab_file, codes_desc=None):
    #reads vocab_file into two lookups (word:ind) and (ind:word)
    vocab = set()
    
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    if codes_desc is not None:
        desc = set([w for d in codes_desc for w in d])
        vocab = vocab.union(desc)
    ind2w = {i+1:w for i,w in enumerate(sorted(vocab))}
    w2ind = {w:i for i,w in ind2w.items()}
    
    return ind2w, w2ind


def load_full_codes(train_path, hier=False):
    """
        Inputs:
            train_path: path to train dataset
        Outputs:
            code lookup, description lookup
    """
    codes = set()
    codes_train_examples = Counter()
    codes_test_examples = Counter()
    codes_rank = Counter()
    
    #build code lookups from appropriate datasets
    for split in ['train', 'dev', 'test']:
        if train_path.endswith('.ndjson'):
            with jsonlines.open(train_path.replace('train', split), 'r') as f:
                for row in f:
                    codes.update([c.strip('.') for c in row[4]])
                    if split == 'train':
                        codes_train_examples.update([c.strip('.') for c in row[4]])
                    elif split == 'test':
                        codes_test_examples.update([c.strip('.') for c in row[4]])
                    codes_rank.update([c.strip('.') for c in row[4]])
        else:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    codes.update([c.strip('.') for c in row[3].split(';')])
                    if split == 'train':
                        codes_train_examples.update([c.strip('.') for c in row[3].split(';')])
                    elif split == 'test':
                        codes_test_examples.update([c.strip('.') for c in row[3].split(';')])
                    codes_rank.update([c.strip('.') for c in row[3].split(';')])

    codes = set([str(c) for c in codes if c != ''])
    codes_train_examples = {code : count for code, count in codes_train_examples.most_common()}
    codes_test_examples = {code : count for code, count in codes_test_examples.most_common()}
    codes_rank = {code : rank+1 for rank, (code, _) in enumerate(codes_rank.most_common())}
    
    return codes, codes_train_examples, codes_test_examples, codes_rank
    

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    if code == '':
        return code
        
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:] if len(code) > 4 else code
        else:
            code = code[:3] + '.' + code[3:] if len(code) > 3 else code
    else:
        code = code[:2] + '.' + code[2:] if len(code) > 2 else code
    return code

def load_code_descriptions(data_dir, exclude_non_billable = False):
    #load description lookup from the appropriate data files
    desc_dict_plain = defaultdict(str)
    codes_billable = {}
    
    with open(os.path.join(data_dir, 'D_ICD_DIAGNOSES.csv'), 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = reformat(row[1], True)
            desc = row[-1]
            desc_dict_plain[code] = [word.lower() for word in re.split('\W+', desc) if word.isalpha()]
            codes_billable[code] = True

    with open(os.path.join(data_dir, 'D_ICD_PROCEDURES.csv'), 'r') as descfile:
        r = csv.reader(descfile)
        #header
        next(r)
        for row in r:
            code = reformat(row[1], False)
            desc = row[-1]
            desc_dict_plain[code] = [word.lower() for word in re.split('\W+', desc) if word.isalpha()]
            codes_billable[code] = True

    if not exclude_non_billable:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ICD9_descriptions'), 'r') as labelfile:
            for i,row in enumerate(labelfile):
                row = row.rstrip().split()
                code = str(row[0])
                
                if code not in desc_dict_plain.keys():
                    desc = ' '.join(row[1:])
                    desc_dict_plain[code] = [word.lower() for word in re.split('\W+', desc) if word.isalpha()]
                    codes_billable[code] = False

    return desc_dict_plain, codes_billable
    

class MimicDataset(Dataset):

    def __init__(self, filename, dicts, num_labels_fine, num_labels_coarse, max_len=-1):
    
        print('loading data from ' + filename)
        
        ind2w, w2ind, ind2c, c2ind, ind2c_coarse, c2ind_coarse = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['ind2c_coarse'], dicts['c2ind_coarse']
    
        if filename.endswith('.csv'):
            self.notes_labeled = pd.read_csv(filename, dtype={0:np.int32, 1:np.int32, 3:str, 4:np.int32})
            self.notes_labeled.columns = ['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS', 'LENGTH']
            self.notes_labeled['LABELS'] = self.notes_labeled['LABELS'].apply(lambda labels : list(str(labels).split(';')))
            self.notes_labeled['TEXT'] = self.notes_labeled['TEXT'].apply(lambda text : text.split(' '))

        elif filename.endswith('.ndjson'):
            self.notes_labeled = pd.read_json(filename, lines=True, dtype={0:np.int32, 1:np.int32, 3:np.int32})
            self.notes_labeled.columns = ['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LENGTH', 'LABELS']
            self.notes_labeled['TEXT'] = self.notes_labeled['TEXT'].apply(lambda text : [word for sentence in text for word in sentence])
        else:
            raise ValueError('dataset file must have .csv or .ndjson extension')

        self.num_labels_fine = num_labels_fine
        self.num_labels_coarse = num_labels_coarse
        self.max_len = max_len
        
        self.notes_labeled['TEXT_PLAIN'] = self.notes_labeled['TEXT']
        self.notes_labeled['TEXT'] = self.notes_labeled['TEXT'].apply(lambda text : np.array([int(w2ind.get(word) or len(w2ind)+1) for word in text], dtype=np.int32))
        self.notes_labeled['LABELS_COARSE'] = self.notes_labeled['LABELS'].apply(lambda labels : np.array(list(set([idx for idx in ( int(c2ind_coarse.get(str(l).split('.')[0], -1)) for l in labels ) if idx != -1])), dtype=np.int32))
        self.notes_labeled['LABELS'] = self.notes_labeled['LABELS'].apply(lambda labels : np.array([idx for idx in ( int(c2ind.get(str(l), -1)) for l in labels ) if idx != -1], dtype=np.int32))
        
    def __len__(self):
        return len(self.notes_labeled)

    def __getitem__(self, idx):
        item = self.notes_labeled.iloc[[idx]]

        data = item['TEXT'].values[0]

        if self.max_len > -1 and len(data) >= self.max_len:
            data = data[:self.max_len]
        data = torch.LongTensor(data)

        labels_idx_fine = item['LABELS'].values[0]
        target_fine = np.zeros(self.num_labels_fine)
        target_fine[labels_idx_fine] = 1
        target_fine = torch.FloatTensor(target_fine)

        labels_idx_coarse = item['LABELS_COARSE'].values[0]
        target_coarse = np.zeros(self.num_labels_coarse)
        target_coarse[labels_idx_coarse] = 1
        target_coarse = torch.FloatTensor(target_coarse)

        hadm_id = item['HADM_ID'].values[0]
        doc = item['TEXT_PLAIN'].values[0]
        
        return data, target_fine, target_coarse, hadm_id, doc

def collate(batch):

    data, target_fine, target_coarse, hadm_ids, docs = zip(*batch)

    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    
    target_fine = torch.stack(target_fine)
    target_coarse = torch.stack(target_coarse)

    return data, target_fine, target_coarse, hadm_ids, docs
