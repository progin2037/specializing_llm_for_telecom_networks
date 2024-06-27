import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import remove_release_number, llm_inference, get_results_with_labels, create_empty_directory

torch.set_default_device("cuda")

MODEL_USED = 'Phi-2'
USE_REPO_MODEL = True  # use fine-tuned model from repo (models/peft_phi_2_repo)
USE_LOCAL_FINE_TUNED = False  # use own fine-tuned model (models/peft_phi_2)
USE_MODEL_FROM_HUGGINGFACE = False  # use the original model without fine-tuning
DO_TRAIN_INFERENCE = False  # do train inference (True) or only test inference (False)

if USE_REPO_MODEL:
    model_path = 'models/peft_phi_2_repo'
elif USE_LOCAL_FINE_TUNED:
    model_path = 'models/peft_phi_2'
else:
    model_path = 'microsoft/phi-2'

# Read PHI-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype="auto",
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2',
                                          trust_remote_code=True)

# Read data
train = pd.read_json('data/TeleQnA_training.txt').T
labels = pd.read_csv('data/Q_A_ID_training.csv')
test = pd.read_json('data/TeleQnA_testing1.txt').T
test_new = pd.read_json('data/questions_new.txt').T
# Merge test with additional questions from test_new
test = pd.concat([test, test_new])

# Create question ID column (question number)
train['Question_ID'] = train.index.str.split(' ').str[-1]
test['Question_ID'] = test.index.str.split(' ').str[-1]
# Remove [3GPP Release <number>] from question
train = remove_release_number(train, 'question')
test = remove_release_number(test, 'question')

# Preparation for output saving
create_empty_directory('results')
today_date = pd.to_datetime('today').strftime('%Y_%m_%d')

# Train data Inference
if DO_TRAIN_INFERENCE:
    # Train data inference
    results_train, _ = llm_inference(train, model, tokenizer)
    results_labels, train_acc = get_results_with_labels(results_train, labels)
    # Save train results
    results_labels.to_csv(f'results/{today_date}_{MODEL_USED}_train_results.csv', index=False)

# Test data inference
results_test, _ = llm_inference(test, model, tokenizer)
results_test = results_test.astype('int')
results_test['Task'] = MODEL_USED
# Save test results
results_test.to_csv(f'results/{today_date}_{MODEL_USED}_test_results.csv', index=False)
