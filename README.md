### DILATED CONVOLUTION NETWORKS FOR CLASSIFICATION OF ICD-9 BASED CLINICAL SUMMARIES

This repository contains source code for thesis work: https://webthesis.biblio.polito.it/14400/

## Dependencies used, You can also use the new available versions.

* Python 3.6
* pytorch 0.4.1
* tqdm
* scikit-learn 0.19.1
* numpy 1.13.3, scipy 0.19.1, pandas 0.24.1
* jsonlines

It is better to create a new enviroment in order to avoid loosing update versions of some packages. Install Virtual ENV using ``pip install virtualenv `` then create an enviroment using `` virtualenv new  `` and ``source new/bin/activate``
you can simply use the `` pip install -r req.text`` to use the packages in a new virtual enviroment. 


## For Training a new model

Create a directory that holds the uncompressed files `D_ICD_DIAGNOSES.csv` and `D_ICD_PROCEDURES.csv` from your MIMIC-III database copy and a ```trained``` folder that will hold your trained models.

To train a new model, use the script `training.py`. Execute `python training.py -h` for a full list of input arguments and flags.

Use the following files as input for the model. This data is pre-processed by "Stefano Malacrino". 
## Preprocessed data format 

Columns:0     1                 2                   3               4
PatientID; HospitalID; Lists_of_Text_Tokens; Length_of_lists; List_of_Labels



* `notes_train.ndjson` training split of the dataset
* `notes_dev.ndjson` development split of the dataset
* `notes_test.ndjson` validation split of the dataset
* `Word2Vec Pretrained Embeddings` can be downloaed from (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
* `glove.42B.300d.txt` GloVe pre-trained word vectors (available at https://nlp.stanford.edu/data/glove.42B.300d.zip)
* `vocab.csv` vocabulary of the training corpus


## To train a new model run the following command

```python3 main.py /path/to/notes_train.ndjson /path/to/vocab.csv 300,125,125,125,125,125 --embed glove --filter-size 3 --dilation 1,2,4,8,16 --dropout 0.15,0.15,0.15,0.15,0.15 --n-epochs 100 --lr 0.001 --criterion f1_micro_fine --patience 5 --batch-size 8 --max-len 5000 --embed-file /path/to/glove.42B.300d.txt --embed-desc --models-dir /path/to/trained --data-dir /path/to/directory```

if you want to use gpu, add --gpu in the commands

After Training, If you want to test a model use the following command 

```python3 main.py /path/to/notes_train.ndjson /path/to/vocab.csv 300,125,125,125,125,125 --filter-size 3 --dilation 1,2,4,8,16 --dropout 0.15,0.15,0.15,0.15,0.15 --n-epochs 100 --criterion f1_micro_fine --patience 5 --lr 0.001--batch-size 8 --max-len 5000 --embed-file /path/to/glove.42B.300d.txt --embed-desc --models-dir /path/to/trained --data-dir /path/to/directory --test-model /path/to/Optim_last_epoch.pth```

use 

```python3 main.py -h``` to observe full range of paramteric values. 

## Parametric Tuning

The Change in filter size and dropout probabilty results in variation of evaluation score. It can be managed to certain range based on targeted criteria. Here is an excerpt from change that underlies from filter size [3,4,5,6] & Dropout [0.1,0.15,0.2] with FastText.


| <img src="https://funkyimg.com/i/31kPs.png" width="350"> | <img src="https://funkyimg.com/i/31kPD.png" width="350"> 

If you use this code, then please cite this.
`
@phdthesis{kanwal2020dilated,
  title={Dilated Convolution Networks for Classification of ICD-9 based Clinical Summaries},
  author={Kanwal, Neel},
  year={2020},
  school={Politecnico di Torino}
}
`
