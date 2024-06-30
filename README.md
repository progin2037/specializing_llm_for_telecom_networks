# specializing_llm_for_telecom_networks
This repository contains processing for "Specializing Large Language Models for Telecom Networks by ITU AI/ML in 5G
Challenge" (https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks).
The solution is created in Python, using Microsoft's Phi-2 model (https://huggingface.co/microsoft/phi-2).
It contains prompt engineering, simple RAG application and model fine-tuning.

## How to run
The solution was created using Python version 3.10.13 on Windows 11.

1. Clone this repo
2. Install packages from requirements (`pip install -r requirements.txt`)
3. Download data
   1. Join the competition (https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks)
   2. Download competition data and copy it to data/ directory inside your cloned repository
   3. Extract rel18 folder from rel18.rar
5. [Optional] Run vectore_store_for_rag.py
    * It is possible to run model without using RAG (by changing PERFORM_RAG to False in main.py)
    * This RAG implementation uses rel18/ documents from the competition
    * It is possible to extract data only from some portion of the documents if running into memory issues. Set
 SAMPLE_DOCS in vectore_store_for_rag.py to True and set your SAMPLE_FRAC (a fraction of documents to retrieve)
6. [Optional] Run fine_tuning.py
   * Already fine-tuned model is also available in models/peft_phi_2_repo/. Keep in mind that the model from repo was
 fine-tuned with RAG context.
   * Keep in mind that your own fine-tuned model won't be exactly the same as the one from repo
   * You could decide if you want to use fine-tuning with context from RAG. Default is fine-tuning with RAG context. To
 change it, set USE_RAG to False in fine_tuning.py
7. Run main.py
   * There are 3 options of models to load: use model from repo (USE_REPO_MODEL), use own fine-tuned model
 (USE_LOCAL_FINE_TUNED) and USE_MODEL_FROM_HUGGINGFACE. USE_REPO_MODEL is set to True on default. To use different
 model, set the relevant option to True and other options to False
