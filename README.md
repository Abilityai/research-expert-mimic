# expert-mimic
Experiment with expert-mimic via QLoRA on interviews' data

This repository contains the code for the paper "Experiment on Fact Memorization in LLM 
Using QLoRA and Quality Assessment of Such Memorization"

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

In addition, you need to install the unsloth package. Its installation depends on 
the hardware you are using. Please refer to the 
[official documentation](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions)
for more information.


This code was tested with Python 3.11 and two GPUs: NVIDIA GeForce RTX 3090 on the 
local Ubuntu machine and L4 on the [LightningStudio](https://lightning.ai/) platform.

Before running the code, you also need to fill the '.env' file with the necessary
environment variables. You can use the '.env.example' file as a template.

## Usage
### Datasets

* [ChatML dataset](data/chat) - a dataset of interviews with chatbots. Files starting 
with an underscore should be treated as a test set.
* [Fact Q/A dataset](data/fact_qa) - a dataset with facts and questions about them 
for the three newest interviews (those containing "2023" in the filename).


### Style experiment
To perform the style experiment, we need to make 3 steps:

1. Train the LLM model on the ChatML dataset:
```bash
python3 style_experiment/us_finetune_musk.py
```
This script will train the LLM models and push the results to the Hugging Face Hub.

2. Generate the responses for the test set:
```bash
python3 style_experiment/predict_test_dataset.py
```
This script will generate the responses for the test set and save them to the
`style_experiment/data/musk/*.jsonl` files.

3. To evaluate the responses (pairwise comparison), run the following command for 
every pair you want to compare:
```bash
python3 style_calculator.py {path_to_file_a.jsonl} {path_to_file_b.jsonl}
```
This script generates the reports for the pairwise comparison of the responses. This
report contains two files:
- CSV with per-sample results for debugging
- TXT with the final result

Reports are saved to the folder of the first file with the name pattern 
`{file_a_name}_vs_{file_b_name}.*`

### Fact memorization experiment
To perform the fact memorization experiment, we need to make 3 steps:

1. Train the LLM model on the Fact Q/A dataset:
```bash
python3 fact_memo_experiment/us_finetune_musk.py
```
This script will train the LLM models and push the results to the Hugging Face Hub.

2. Generate the responses for the Fact Q/A dataset: 
```bash
python3 fact_memo_experiment/predict_qa_dataset.py
```

3. To evaluate the responses (pairwise comparison), run the following command for
every pair you want to compare:
```bash
python3 facts_calculator.py {path_to_file_a.jsonl} {path_to_file_b.jsonl}
```
This script generates the reports for the pairwise comparison of the responses. This
report contains two files with the same names as in the style experiment. However, the 
reports contain different metrics as described in the paper.

### Models
The trained models are available in the Hugging Face Hub:
https://huggingface.co/atepeq

These names are hardcoded in the prediction scripts (see the second step in the
style and fact memorization experiments).

If you want to reproduce the training process, you'll need to generate your own
Huggingface API token and replace the `HUGGINGFACE_TOKEN` in the `.env` file. In 
addition, you'll need to change full model names in the scripts.
