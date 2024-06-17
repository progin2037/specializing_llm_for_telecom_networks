import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import llm_inference, create_empty_directory

torch.set_default_device("cuda")

# Read PHI-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                             torch_dtype="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2",
                                          trust_remote_code=True)

# Read train data
train = pd.read_json('data/TeleQnA_training.txt').T
# Read labels
labels = pd.read_csv('data/Q_A_ID_training.csv')
# Read test data
test = pd.read_json('data/TeleQnA_testing1.txt').T

# Get question ID column (a number of the question)
train['question_id'] = train.index.str.split(' ').str[-1]
test['question_id'] = test.index.str.split(' ').str[-1]

# Train data inference
results_train, _ = llm_inference(train, model, tokenizer)
# Change column name
results_train.rename({'Answer_ID': 'Prediction_ID'}, axis=1, inplace=True)
# Remove empty columns from labels
labels = labels[['Question_ID', 'Answer_ID']]
# Set question IDs column to the same format as in labels
results_train['Question_ID'] = results_train['Question_ID'].astype('int')
# Merge columns
results_labels = pd.merge(labels,
                          results_train,
                          how='left',
                          on='Question_ID')
# Get accuracy of predictions
train_acc = 100 * (results_labels['Answer_ID'] == results_labels['Prediction_ID']).sum() / len(results_labels)
print(f'Train accuracy: {train_acc}%')
# Save train results
create_empty_directory('results')
today_date = pd.to_datetime('today').strftime('%Y_%m_%d')
model_used = 'Phi-2'
results_labels.to_csv(f'results/{today_date}_{model_used}_train_results.csv', index=False)

# Test data inference
results_test, _ = llm_inference(test, model, tokenizer)
results_test = results_test.astype('int')
results_test['Task'] = model_used
# Save results
results_test.to_csv(f'results/{today_date}_{model_used}_test_results.csv', index=False)
