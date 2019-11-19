### Deep Dilated Convolution Neural Networks

## Dependencies used, You can also use the new available versions.

* Python 3.6
* pytorch 0.4.1
* tqdm
* scikit-learn 0.19.1
* numpy 1.13.3, scipy 0.19.1, pandas 0.24.1
* jsonlines



## For Training a new model

Create a directory that holds uncompressed files `D_ICD_DIAGNOSES.csv` and `D_ICD_PROCEDURES.csv` from your MIMIC-III database copy and a ```trained``` folder that will hold your trained models.

To train a new model use the script `training.py`. Execute `python training.py -h` for a full list of input arguments and flags.

Use the following files as input for the model, This data is pre-processed by "Stefano Malacrino". 

* `notes_train.ndjson` training split of the dataset
* `notes_dev.ndjson` development split of the dataset
* `notes_test.ndjson` validation split of the dataset
* `glove.840B.300d.txt` GloVe pre-trained word vectors (available at https://nlp.stanford.edu/data/glove.840B.300d.zip)
* `vocab.csv` vocabulary of the training corpus


## To train a new model run the following command

```python3 main.py /path/to/notes_train.ndjson /path/to/vocab.csv full deep_dilated 300,125,125,125,125,125 --filter-size 4 --dilation 1,2,4,8,16 --dropout 0.2,0.2,0.2,0.2,0.2 --n-epochs 100 --lr 0.001 --criterion f1_micro_fine --patience 5 --batch-size 8 --max-len 5200 --embed-file /path/to/glove.840B.300d.txt --embed-desc --models-dir /path/to/trained --data-dir /path/to/directory```

if you want to use gpu, add --gpu in the commands

After Training, If you want to test a model use the following command 

```python3 main.py /path/to/notes_train.ndjson /path/to/vocab.csv full deep_dilated 300,125,125,125,125,125 --filter-size 4 --dilation 1,2,4,8,16 --dropout 0.2,0.2,0.2,0.2,0.2 --n-epochs 100 --criterion f1_micro_fine --patience 5 --lr 0.001--batch-size 8 --max-len 5200 --embed-file /path/to/glove.840B.300d.txt --embed-desc --models-dir /path/to/trained --data-dir /path/to/directory --test-model /path/to/Optim_last_epoch.pth```
