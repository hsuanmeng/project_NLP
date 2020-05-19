import pandas as pd
import numpy as np
import re
import json
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import argparse

# Importing BERT modules
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

# Importing common modules
from common import ModelCommonUse

def argsparse():
    # Initiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input csvfile")

    # Read arguments from the command line
    args = parser.parse_args()
    return args

# TODO: 改成讀資料的路徑（預計是用最新日期抓）
args = argsparse()
if args.input:
    df = pd.read_csv(args.input)
origin_df = pd.read_csv("./data/chat_alldata.csv")
origin_df = origin_df.append(df)
origin_df.to_csv("./data/chat_alldata.csv", index=False, encoding="utf-8-sig")

# 把意圖是A的拿掉
origin_df = origin_df[origin_df["意圖"]!="A"]
origin_df = origin_df[~origin_df["內容"].isna()]

# 測試分類數量的準確度所做的準備，看哪裡降很多
intent_count = pd.DataFrame(origin_df["意圖"].value_counts())
classcount = 620
intent_categroy_df = intent_count[intent_count["意圖"] >= classcount ]

# 篩選出筆數夠多的意圖，並將意圖編號
num = 0
newDF = pd.DataFrame()
for content in intent_categroy_df.index.to_list():
    SelDF = origin_df[origin_df["意圖"]==content]
    SelDF["Section"] = num
    SelDF = shuffle(SelDF, random_state=100)
    newDF = newDF.append(SelDF)
    num+=1
newDF = newDF.reset_index(drop=True)

# 切分 訓練集 跟 測試集
TrainDF, TestDF = pd.DataFrame(), pd.DataFrame()

for i in range(len(intent_categroy_df)):
    trainDF = newDF[newDF["Section"]==i]
    trainDF, testDF =  train_test_split(trainDF, test_size = 0.1, random_state = 100)
    TrainDF = TrainDF.append(trainDF)
    TestDF = TestDF.append(testDF)

# 將訓練集再進行切分，1/3 做 validation，2/3 做 training
TtrainDF, ValidDF = pd.DataFrame(), pd.DataFrame()

for i in range(len(intent_categroy_df)):
    TotrainDF = TrainDF[TrainDF["Section"]==i]
    TotrainDF, TovalidDF = train_test_split(TotrainDF, test_size=1/3, random_state = 100)
    TtrainDF = TtrainDF.append(TotrainDF)
    ValidDF = ValidDF.append(TovalidDF)

CutDF_train = TtrainDF[["內容", "Section"]].reset_index(drop=True)
CutDF_valid = ValidDF[["內容", "Section"]].reset_index(drop=True)

# 要訓練的欄位跟分類欄位
DATA_COLUMN = "內容"
LABEL_COLUMN = "Section"
label_list = list(CutDF_train['Section'].unique())

# Set the output directory for saving model file
OUTPUT_DIR = "./bert_news_category_WithChatdata_10v5"

#@markdown Whether or not to clear/delete the directory and create a new one
DO_DELETE = False
if DO_DELETE:
    try:
        tf.gfile.DeleteRecursively(OUTPUT_DIR)
    except:
        pass
tf.gfile.MakeDirs(OUTPUT_DIR)

train_InputExamples = CutDF_train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

val_InputExamples = CutDF_valid.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

tokenizer = ModelCommonUse().create_tokenizer_from_hub_module()

# load steps from jsonfile
steps_dict = json.load(open("train_steps.json"))
# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = steps_dict["MAX_SEQ_LENGTH"]

# Convert our train and validation features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
val_features = bert.run_classifier.convert_examples_to_features(val_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

steps_dict["train_features"] = len(train_features)
steps_dict["label_list"] = list(newDF["意圖"].unique())

with open("train_steps.json", "w") as jsonfile:
    json.dump(steps_dict, jsonfile)

# Compute train and warmup steps from batch size
BATCH_SIZE = steps_dict["BATCH_SIZE"]
LEARNING_RATE = steps_dict["LEARNING_RATE"]
NUM_TRAIN_EPOCHS = steps_dict["NUM_TRAIN_EPOCHS"]
# Warmup is a period of time where the learning rate is small and gradually increases--usually helps training.
WARMUP_PROPORTION = steps_dict["WARMUP_PROPORTION"]
# Model configs
SAVE_CHECKPOINTS_STEPS = steps_dict["SAVE_CHECKPOINTS_STEPS"]
SAVE_SUMMARY_STEPS = steps_dict["SAVE_SUMMARY_STEPS"]

# Compute train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify output directory and number of checkpoint steps to save
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

#Initializing the model and the estimator
model_fn = ModelCommonUse().model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)

# Create an input function for validating. drop_remainder = True for using TPUs.
val_input_fn = run_classifier.input_fn_builder(
    features=val_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

# Training the model
print(f"Beginning Training!")
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

#Evaluating the model with Validation set
print("Validation:", estimator.evaluate(input_fn=val_input_fn, steps=None))
