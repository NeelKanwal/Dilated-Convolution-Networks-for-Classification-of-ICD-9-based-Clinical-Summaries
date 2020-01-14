import argparse
import os 
import numpy as np
import sys
import time
import json
import sister
import gensim #for Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import datasets

def main(args, reporter=None):
    start = time.time()
    print("loading lookups...")
    dicts = datasets.load_lookups(args, hier=args.hier)
    if args.embed == "glove":
        print("Embedding with Glove......")
        word_embeddings_matrix = load_glove_embeddings(args.embed_file1, dicts['ind2w'], args.dims[0], args.embed_normalize)
    elif args.embed == "word2vec":
        print("Embedding with Word2Vec......")
        word_embeddings_matrix = load_word2vec_embeddings(args.embed_file1, dicts['ind2w'], args.dims[0], args.embed_normalize)
    elif args.embed == "stack_word2vec":
        print("Embedding with Stack of GloVe & Word2Vec......")
        word_embeddings_matrix = load_stack_word2vec_embeddings(args.embed_file1,args.embed_file2, dicts['ind2w'], args.dims[0], args.embed_normalize)
    elif args.embed == "stack_fasttext":
        print("Embedding with Stack of GloVe & fasttext......")
        word_embeddings_matrix = load_stack_fasttext_embeddings(args.embed_file1, dicts['ind2w'], args.dims[0], args.embed_normalize)    
    else:
        print("Making fasttext Embeddings")
        word_embeddings_matrix = load_fasttext_Embeddings(dicts['ind2w'], args.embed_normalize)

    elapsed = round(time.time() - start)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print("TOTAL ELAPSED TIME: {:d}:{:02d}:{:02d}".format( h, m, s))
    return word_embeddings_matrix
    
def load_stack_fasttext_embeddings(embed_file1, ind2w, embed_size, embed_normalize):
    word_embeddings = {}
    vocab_size = len(ind2w)
    print(vocab_size)
    W = np.zeros((vocab_size+2, embed_size))
    words_found_glove = 0
    words_found_ftt = 0
    words_found = 0

    embedder = sister.MeanEmbedding(lang="en")
    
    with open(embed_file1) as ef:
        for line in ef:
            line = line.rstrip().split()
            idx = len(line) - embed_size
            word = '_'.join(line[:idx]).lower().strip()
            vec = np.array(line[idx:]).astype(np.float)
            word_embeddings[word] = vec
     
    for ind, word in ind2w.items():
        try:
            try: 
              W[ind] = word_embeddings[word]                  
              words_found_glove += 1
              words_found += 1         
            except KeyError:
              W[ind] = embedder(word) 
              words_found_ftt += 1
              words_found += 1
        except:
            W[ind] = np.random.randn(1, embed_size)  
        if embed_normalize:
            W[ind] = W[ind] / (np.linalg.norm(W[ind]) + 1e-6)

    W[vocab_size-1] = np.random.randn(1, embed_size)
    
    if embed_normalize:
        W[vocab_size-1] = W[vocab_size-1] / (np.linalg.norm(W[vocab_size-1]) + 1e-6)
    print('GloVe vocabulary coverage: {}'.format(words_found_glove/vocab_size))
    print('FastText vocabulary coverage: {}'.format(words_found_ftt/vocab_size))
    print('Total vocabulary coverage: {}'.format(words_found/vocab_size)) 
    return W 

def load_fasttext_Embeddings (ind2w, normalize):
    #"""load FastText model directly from SISTER """

    embedderr = sister.MeanEmbedding(lang="en")
    word_embeddings = {}
    vocab_size = len(ind2w)
    W = np.zeros((vocab_size+2, 300))
    words_found= 0

    for ind, word in ind2w.items():
        try: 
            W[ind] = embedderr(word)                  
            words_found += 1         
            
        except:
            W[ind] = np.random.randn(1, 300)  
        if normalize:
            W[ind] = W[ind] / (np.linalg.norm(W[ind]) + 1e-6)

    W[vocab_size-1] = np.random.randn(1, 300)
    
    if normalize:
        W[vocab_size-1] = W[vocab_size-1] / (np.linalg.norm(W[vocab_size-1]) + 1e-6)
    print('Total vocabulary coverage: {}'.format(words_found/vocab_size)) 
    return W 

def load_stack_word2vec_embeddings(embed_file1,embed_file2, ind2w, embed_size, embed_normalize):
   # Please add location to section embeddings for this function
    word_embeddings = {}
    vocab_size = len(ind2w)
    print(vocab_size)
    W = np.zeros((vocab_size+2, embed_size))
    words_found_glove = 0
    words_found_vec = 0
    words_found = 0
    
    wv_embeddings = KeyedVectors.load_word2vec_format(embed_file2,binary=True) 
    with open(embed_file1) as ef:
        for line in ef:
            line = line.rstrip().split()
            idx = len(line) - embed_size
            word = '_'.join(line[:idx]).lower().strip()
            vec = np.array(line[idx:]).astype(np.float)
            word_embeddings[word] = vec
     
    for ind, word in ind2w.items():
        try:
            try: 
              W[ind] = word_embeddings[word]                  
              words_found_glove += 1
              words_found += 1         
            except KeyError:
              W[ind] = wv_embeddings[word] 
              words_found_vec += 1
              words_found += 1
        except:
            W[ind] = np.random.randn(1, embed_size)  
        if embed_normalize:
            W[ind] = W[ind] / (np.linalg.norm(W[ind]) + 1e-6)

    W[vocab_size-1] = np.random.randn(1, embed_size)
    
    if embed_normalize:
        W[vocab_size-1] = W[vocab_size-1] / (np.linalg.norm(W[vocab_size-1]) + 1e-6)
    print('GloVe vocabulary coverage: {}'.format(words_found_glove/vocab_size))
    print('Word2Vec vocabulary coverage: {}'.format(words_found_vec/vocab_size))
    print('Total vocabulary coverage: {}'.format(words_found/vocab_size)) 
    return W 

def load_word2vec_embeddings(embed_file1,ind2w, embed_size, embed_normalize):

    vocab_size = len(ind2w)
    W = np.zeros((vocab_size+2, embed_size))
    words_found = 0
    wv_embeddings = KeyedVectors.load_word2vec_format(embed_file1,binary=True) 
    for ind, word in ind2w.items():

        try: 
            W[ind] = wv_embeddings[word]
            words_found += 1
        except KeyError:
            W[ind] = np.random.randn(1, embed_size)
        if embed_normalize:
            W[ind] = W[ind] / (np.linalg.norm(W[ind]) + 1e-6)

    W[vocab_size-1] = np.random.randn(1, embed_size)
    
    if embed_normalize:
        W[vocab_size-1] = W[vocab_size-1] / (np.linalg.norm(W[vocab_size-1]) + 1e-6)

    print('vocabulary coverage: {}'.format(words_found/vocab_size))
    return W 
           
def load_glove_embeddings(embed_file1, ind2w, embed_size, embed_normalize):
    word_embeddings = {}
    vocab_size = len(ind2w)
    
    with open(embed_file1) as ef:
        for line in ef:
            line = line.rstrip().split()
            idx = len(line) - embed_size
            word = '_'.join(line[:idx]).lower().strip()
            vec = np.array(line[idx:]).astype(np.float)
            word_embeddings[word] = vec
    W = np.zeros((vocab_size+2, embed_size))
    words_found = 0
    
    for ind, word in ind2w.items():

        try: 
            W[ind] = word_embeddings[word]
            words_found += 1
        except KeyError:
            W[ind] = np.random.randn(1, embed_size)
        if embed_normalize:
            W[ind] = W[ind] / (np.linalg.norm(W[ind]) + 1e-6)

    W[vocab_size-1] = np.random.randn(1, embed_size) 
    if embed_normalize:
        W[vocab_size-1] = W[vocab_size-1] / (np.linalg.norm(W[vocab_size-1]) + 1e-6)

    print('vocabulary coverage: {}'.format(words_found/vocab_size))
    
    return W 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To find the Coverage from existing Model")
    parser.add_argument("data_path", type=str,
                        help="path to a file containing train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("vocab", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("Y", type=str, help="size of label space")
    parser.add_argument("dims", type=lambda s: [int(dim) for dim in s.split(',')], help="layers dimensions")
    parser.add_argument("--embed", type=str, choices=["glove", "stack_word2vec","stack_fasttext", "word2vec","fasttext"] , required=False, dest="embed", default='fasttext',help="Choose a type of word Embedding layer")
    parser.add_argument("--embed-file1", type=str, required=False, dest="embed_file1",
                        help="path to a file holding pre-trained embeddings [GloVe,word2Vec]")
    parser.add_argument("--embed-file2", type=str, required=False, dest="embed_file2",
                        help="path to a file if using Stack")
    parser.add_argument("--embed-normalize", action='store_true', dest="embed_normalize",
                        help="optional flag to normalize word embeddings (defaul false)")
    parser.add_argument("--data-dir", type=str, dest="data_dir", required=True, help="path to mimic data directory")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available (defaul false)")
    parser.add_argument("--hier", action="store_true", dest="hier",
                        help="hierarchical predictions (defaul false)")
    parser.add_argument("--exclude-non-billable", action="store_true", dest="exclude_non_billable", help= "(defaul false)")
    parser.add_argument("--include-invalid", action="store_true", dest="include_invalid", help= "(defaul false)")
    parser.add_argument("--embed-desc", action="store_true", dest="embed_desc", help= "(defaul false)")
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)

