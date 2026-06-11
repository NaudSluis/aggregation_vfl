

## Codebase explaination

#### Requirements
The required packages are supposed to be in requirements.txt

#### Preprocessing & data
There are two preprocessing files. `data_pre_pandas.py` is the one used to generate the latest version of the dataset.  
This dataset can be loaded from the file `final_lstm_dataset_merged.parquet`.

#### Central DeepAR model
The central DeepAR model architecture is based on two files. The file `central_deepar.py` contains classes used in the training file 
`train_central.py`. In the training file, main is called when ran, which initialises all encoders, model and optimizers and passes them to the run_many() function. That function runs a given amount of epochs and handles validation. For training, it calls train_one_epoch(), which trains one epoch. Plotting functions can be found in `plots_metric_utils.py`.

#### VFL DeepAR 
The files for the VFL DeepAR are `vfl_deepar.py` and `train_vfl.py`. The logic is (supposed to) be the same as in the central version, accept with three seperate encoders for the different clients.

#### VFL DeepAR with attention
The files for the VFL DeepAR are `vfl_aggregation.py`, `vfl_deepar.py` and `train_vfl.py`. `vfl_aggregation.py` contains a class that calculates the attention scores. However, this model is probably still flawed

#### VFL DeepAR with ICAFS
The files for the VFL DeepAR are `vfl_aggregation.py`, `vfl_deepar.py` and `train_vfl.py`. `vfl_aggregation.py` contains a class that calculates the is supposed to handle the selectors. However, this model does not work yet.

## How to run a model

Models can be ran by running the following command in the directory:
`python3 train_central.py`

For all parameters, standard values are put in. These can be changed by doing the following in the call; for example when you want the number 
of buildings to be trained on to be 1:
`python3 train_central.py --num-buildings=1`

Here is an example of how a call could look when you would want many arguments to be different:
`python3 train_central.py --num-buildings=10 --context=168 --pred-window=24 --encoder-hidden=96 --emb-dim 64 --model-hidden 256`

In the main functions, all arguments and a small description can be found. 

> [!NOTE]
> The model is supposed to make an output directory containing checkpoints, a csv with metrics and plots. However, this is designed for snellius, so I am not sure whether this will work immediatly. 

## Latest updates
Here I will post some updates over the coming week on the status of the models