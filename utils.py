import pandas as pd
from pathlib import Path
from transformers.models.phi.modeling_phi import PhiForCausalLM
from transformers.models.codegen.tokenization_codegen_fast import CodeGenTokenizerFast


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


def encode_answer(answer: str) -> int:
    """
    Encode letter to corresponding number.

    Args:
        answer (str): Chosen answer (A/B/C/D or E)
    Returns:
        number (int): Number corresponding to the chosen answer
    """
    letter_to_number = {'A': 1,
                        'B': 2,
                        'C': 3,
                        'D': 4,
                        'E': 5}
    number = letter_to_number[answer]
    return number


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
                  model: PhiForCausalLM,
                  tokenizer: CodeGenTokenizerFast,
                  show_prompts=False,
                  store_wrong=False) -> tuple[pd.DataFrame, list]:
    """
    Perform LLM inference.

    Args:
        data (pd.DataFrame): A DataFrame with data about questions and their options
        model (PhiForCausalLM): Model used for inference
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
    for _, question in data.iterrows():
        prompt = generate_prompt(question)
        if show_prompts:
            print(f"\n{question['question_id']}")
            print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        # Generate only one new character. It should be our answer
        outputs = model.generate(**inputs, max_length=inputs[0].__len__()+1, pad_token_id=tokenizer.eos_token_id)
        answer_letter = tokenizer.batch_decode(outputs)[0][len(prompt):len(prompt)+1]
        # Encode letter to number
        try:
            answer = encode_answer(answer_letter)
        except:
            print(f"Question {question['question_id']} output was improper ({answer_letter})! Changing answer to 1...")
            answer = 1
            if store_wrong:
                wrong_format.append([question['question_id'], answer_letter])
        answers.append([question['question_id'], answer])
    # Create a DataFrame with answers
    answers = pd.DataFrame(answers, columns=['Question_ID', 'Answer_ID'])
    return answers, wrong_format


def create_empty_directory(path: str):
    """
    Create an empty directory if it doesn't exist.

    Args:
        path (str): A path to the directory. It is a parent directory to the current working directory.
    """
    models_path = Path(path)
    models_path.mkdir(parents=True, exist_ok=True)
