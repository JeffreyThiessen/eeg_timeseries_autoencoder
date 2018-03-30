# eeg timeseries autoencoder

This time series encoder was adapted form the pytorch sequence to sequence encoder tutorials.
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

### Install
To use this code please install pytorch, and mne

http://pytorch.org

MNE can be installed alongside other useful tools by installing braindecode

https://robintibor.github.io/braindecode/index.html

### Run

Initialize the data with
```
python -i ts_ae.py
```
To train an encoder use the following to train the basic encoder and sequence to sequence encoders respectively.
```
run()
# or
run_seq()
```

If you want to run with different parameters, channels or data, you will need to modify the code.
