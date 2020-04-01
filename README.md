# Chinese Word Segmentation 

State of the art Chinese Word Segmentation with Bi-LSTMs (Ji Ma, Kuzman Ganchev and David Weiss, EMNLP 2018) - (https://aclweb.org/anthology/D18-1529)

## Compatibility

Python3.6.X,&nbsp;&nbsp;Tensorflow 1.12.0

## Notes

In this project, four chinese datasets (AS,CITYU,MSR and PKU) were used to train the deep learning model for chinese word segmentation task. These datasets can be gotten from: http://sighan.cs.uchicago.edu/bakeoff2005/



## For Training

```bash
Run: python3 train.py
```
input_file_path is the path that contains no-space chinese sequence. &nbsp;

label_file_path is the path that contains the chinese sequence labels in BIES format.

## For Preprocessing

```bash
Run: python3 preprocess.py original_file_path input_file_path output_file_path 
```
original_file_path is the file that contains the chinese sequence. &nbsp;

input_file_path is the path to save the no-space chinese sequence. &nbsp;

label_file_path is the path to save the chinese sequence labels in BIES format.

## For Prediction

```bash
Run: python3 predict.py input_path output_path resources_path
```
input_path is the file that contains the no-space chinese sequence. &nbsp;

output_path is the path to save the predictions in BIES format. &nbsp;

resources_path is the path to the saved model. &nbsp;

The saved model and extras can be downloaded from http://bit.ly/2PKGZBg and placed in the resources folder.

## For Scoring

```bash
Run: python3 score.py predicition_file gold_file
```
prediction_file is the file that contains the predicitions in BIES format from previous step. &nbsp;

gold_file is the path to the gold file in BIES format.
