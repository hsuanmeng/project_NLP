import os
import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import run_classifier
import pandas as pd
import numpy as np
import json
from .common import ModelCommonUse
from datetime import datetime


def Run(filename):
  checkpoint_path = "./bert_news_category_WithChatdata_10v5/"
  # checkpoint_dir = os.path.dirname(checkpoint_path)
  tokenizer = ModelCommonUse().create_tokenizer_from_hub_module()

  # load steps from jsonfile
  # TODO: 路徑要改
  steps_dict = json.load(open("/home/csi/project_NLP/model/train_steps.json"))

  BATCH_SIZE = steps_dict["BATCH_SIZE"]
  LEARNING_RATE = steps_dict["LEARNING_RATE"]
  NUM_TRAIN_EPOCHS = steps_dict["NUM_TRAIN_EPOCHS"]
  WARMUP_PROPORTION = steps_dict["WARMUP_PROPORTION"]
  # Model configs
  SAVE_CHECKPOINTS_STEPS = steps_dict["SAVE_CHECKPOINTS_STEPS"]
  SAVE_SUMMARY_STEPS = steps_dict["SAVE_SUMMARY_STEPS"]

  train_features = steps_dict["train_features"]
  # Compute train and warmup steps from batch size
  num_train_steps = int(train_features / BATCH_SIZE * NUM_TRAIN_EPOCHS)
  num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

  # Specify output directory and number of checkpoint steps to save
  run_config = tf.estimator.RunConfig(
      model_dir=checkpoint_path,
      save_summary_steps=SAVE_SUMMARY_STEPS,
      save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
  
  num_of_label = len(steps_dict["label_list"])
  label_list = [i for i in range(num_of_label)]

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

  # A method to get predictions
  def getPrediction(in_sentences):
    #A list to map the actual labels to the predictions
    # 改成自己的lable
      labels = steps_dict["label_list"]

    # Transforming the test data into BERT accepted form
      input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 0) for x in in_sentences] 
    
    # Creating input features for Test data
      input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

    # Predicting the classes 
      predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
      predictions = estimator.predict(predict_input_fn)
      return [(sentence, prediction['probabilities'], prediction['labels'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

  # TODO: 要改讀資料的路徑
  # ./data/chat_alldata.csv
  # current date and time
  now = datetime.now()
  timestamp = datetime.timestamp(now)
  data_path = "/home/csi/project_NLP/data/"
  filepath = data_path + filename
  need_pred_df = pd.read_csv(filepath)
  # if need_pred_df[""]
  pred_sentences = need_pred_df["內容"].to_list()
  MAX_SEQ_LENGTH = steps_dict["MAX_SEQ_LENGTH"]
  print("====start to predit=====")
  predictions = getPrediction(pred_sentences)

  # 存預測資料的結果，要改一下 code
  pred_sentence = []
  enc_labels = []
  act_labels = []
  for i in range(len(predictions)):
      pred_sentence.append(predictions[i][0])
      enc_labels.append(predictions[i][2])
      act_labels.append(predictions[i][3])

  Pred_Data = pd.DataFrame(np.array([pred_sentence, enc_labels, act_labels]).T, columns = ['Content', 'Section', 'Pred_Intent'])
  MappingDF = need_pred_df[["內容", "意圖"]].reset_index(drop=True).copy()
  Pred_Data = Pred_Data.join(MappingDF)
  Pred_Data["Compare"] = np.where(Pred_Data["Pred_Intent"] == Pred_Data["意圖"], True, False)
  Pred_Data.drop_duplicates(subset="Content", inplace=True)
  final_df = need_pred_df.join(Pred_Data.set_index(["內容","意圖"]), on=["內容","意圖"])

  final_df.to_csv("./data/ouput/Intent_Pred"+str(timestamp)+".csv", encoding="utf-8-sig", index=False)
  result = ModelCommonUse().outputjson(final_df, "意圖", filepath)
  return result


# if __name__ == "__main__":
#   print(Run())