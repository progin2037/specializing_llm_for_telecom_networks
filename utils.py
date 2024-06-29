import pandas as pd
import re
from pathlib import Path
import os
import shutil
from tqdm import tqdm
from transformers.models.phi.modeling_phi import PhiForCausalLM
from transformers.models.codegen.tokenization_codegen_fast import CodeGenTokenizerFast
from peft.peft_model import PeftModelForCausalLM


def remove_release_number(data: pd.DataFrame,
                          column: str) -> pd.DataFrame:
    """
    Remove [3GPP Release <number>] from question.

    Args:
        data (pd.DataFrame):  A DataFrame with data about questions and their options
        column (str): A column name to remove release number from
    Returns:
        data (pd.DataFrame): A DataFrame with removed release number
    """
    data[column] = [re.findall('(.*?)(?:\s+\[3GPP Release \d+]|$)', x)[0] for x in data[column]]
    return data


def get_option_5(row: pd.Series) -> str:
    """
    Retrieve option 5 if available. Otherwise, create an empty string.

    Args:
        row (pd.Series): A row with question and options to choose from
    Returns:
        option_5 (str): Text containing option 5. It will be additionally added to the prompt
    """
    option_5 = row['option 5']
    if pd.isna(option_5):
        option_5 = ''
    else:
        option_5 = f'E) {option_5}'
    return option_5


def encode_answer(answer: str | int,
                  encode_letter: bool = True) -> int | str:
    """
    Encode letter to corresponding number or number to letter.

    Args:
        answer (str): Chosen answer (A/B/C/D/E or 1/2/3/4/5)
        encode_letter (bool) Encode letter to number (True) or number to letter (False). Defaults to True
    Returns:
        number (int | str): Number (letter) corresponding to the chosen answer
    """
    letter_to_number = {'A': 1,
                        'B': 2,
                        'C': 3,
                        'D': 4,
                        'E': 5}
    if encode_letter:
        encoded = letter_to_number[answer]
    else:
        number_to_letter = {y: x for x, y in letter_to_number.items()}
        encoded = number_to_letter[answer]
    return encoded


def generate_prompt(row: pd.Series) -> str:
    """
    Generate prompt for the given row. The prompt template is already created.

    Args:
        row (pd.Series): A row with question and options to choose from
    Returns:
        prompt (str): Generated prompt
    """
    prompt = f"""
    Provide a correct answer to a multiple choice question. Use only one option from A, B, C, D or E.
    {row['question']}
    A) {row['option 1']}
    B) {row['option 2']}
    C) {row['option 3']}
    D) {row['option 4']}
    {get_option_5(row)}
    Answer:
    """
    return prompt


def llm_inference(data: pd.DataFrame,
                  model: PhiForCausalLM | PeftModelForCausalLM,
                  tokenizer: CodeGenTokenizerFast,
                  show_prompts: bool = False,
                  store_wrong: bool = False) -> tuple[pd.DataFrame, list]:
    """
    Perform LLM inference.

    Args:
        data (pd.DataFrame): A DataFrame with data about questions and their options
        model (PhiForCausalLM | PeftModelForCausalLM): Model used for inference
        tokenizer (CodeGenTokenizerFast): Model tokenizer
        show_prompts (bool): Show all generated prompts (True) or not (False). Defaults to False.
        store_wrong (bool): Store questions with not allowed answers (True) or not (False). Defaults to False
    Returns:
        answers (pd.DataFrame): A DataFrame with question IDs and answer IDs
        wrong_format (list): Question ID - Answer ID pairs with improper LLM answers
    """
    answers = []
    wrong_format = []
    # Iterate over different rows
    for _, question in tqdm(data.iterrows()):
        prompt = generate_prompt(question)
        if show_prompts:
            print(f"\n{question['Question_ID']}")
            print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        # Generate only one new character. It should be our answer
        outputs = model.generate(**inputs, max_length=inputs[0].__len__()+1, pad_token_id=tokenizer.eos_token_id)
        answer_letter = tokenizer.batch_decode(outputs)[0][len(prompt):len(prompt)+1]
        # Encode letter to number
        try:
            answer = encode_answer(answer_letter)
        except:
            print(f"Question {question['Question_ID']} output was improper ({answer_letter})! Changing answer to 1...")
            answer = 1
            # Generate more characters to check what is created with the model
            outputs = model.generate(**inputs, max_length=inputs[0].__len__()+20, pad_token_id=tokenizer.eos_token_id)
            answer_letter = tokenizer.batch_decode(outputs)[0]
            print(answer_letter)
            if store_wrong:
                wrong_format.append([question['Question_ID'], answer_letter])
        answers.append([question['Question_ID'], answer])
    # Create a DataFrame with answers
    answers = pd.DataFrame(answers, columns=['Question_ID', 'Answer_ID'])
    return answers, wrong_format


def get_results_with_labels(results_df,
                            labels_df) -> tuple[pd.DataFrame, float]:
    """
    Merge results with ground truth labels.

    Args:
        results_df (pd.DataFrame): Inference results
        labels_df (pd.DataFrame): Ground truth labels
    Returns:
        results_labels (pd.DataFrame): Merged results with labels
        train_acc (float): Model accuracy
    """
    # Change column name to be compatible with labels
    results_df.rename({'Answer_ID': 'Prediction_ID'}, axis=1, inplace=True)
    # Remove empty columns from labels
    labels_df = labels_df[['Question_ID', 'Answer_ID']]
    # Transform question ID column to the same format as in labels
    results_df['Question_ID'] = results_df['Question_ID'].astype('int')
    # Merge columns
    results_labels = pd.merge(labels_df,
                              results_df,
                              how='left',
                              on='Question_ID')
    # Get accuracy of predictions
    train_acc = 100 * (results_labels['Answer_ID'] == results_labels['Prediction_ID']).sum() / len(results_labels)
    print(f'Train accuracy: {train_acc}%')
    return results_labels, train_acc


def create_empty_directory(path: str):
    """
    Create an empty directory if it doesn't exist.

    Args:
        path (str): A path to the directory. It is a parent directory to the current working directory.
    """
    models_path = Path(path)
    models_path.mkdir(parents=True, exist_ok=True)


def create_dir_with_sampled_docs(docs_path: str,
                                 sampled_docs_path: str,
                                 sample_frac: float,
                                 create_new: bool = True):
    """
    Create a new directory with sampled documents.

    Args:
        docs_path (str): Documents directory
        sampled_docs_path (str): New directory to store copied documents
        sample_frac (float): Fraction of documents to store with (0, 1> range
        create_new (bool): Remove all files from the directory if it already exists (True) or not (False)
    """
    # Remove files if directory already exists
    if create_new:
        if os.path.exists(sampled_docs_path):
            shutil.rmtree(sampled_docs_path)
    # Create sampled_docs_path directory if it doesn't exist
    create_empty_directory(sampled_docs_path)
    # Get all documents from docs_path
    dir_list = os.listdir(docs_path)
    # Sample documents
    sampled_docs = pd.Series(dir_list).sample(frac=sample_frac, random_state=22)
    # Copy files
    for file_name in sampled_docs:
        shutil.copy(f'{docs_path}/{file_name}', f'{sampled_docs_path}')
