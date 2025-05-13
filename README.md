# HallucinationDetectionEditing

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

## Intro

This is a repo for our hallucination detection and editing work in the finance domain. This repo includes information on synthetic data generation for training and evaluating our fine-tuned model on FinQA+TATQA.


## Overview 
1. [Data Preparation](#step-1-synthetic-data-generation) 
2. [Inference](#Step-2-model-inference)
3. [Detection Evaluation](#fine-grained-detection)
2. [Editing Evaluation](#factscore)

```
## Data Preparation
```

### Step 1: Error Insertion

```bash
cd data_preparation
python insert_errors.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
--api_key {your_openai_key}
```

### Step 2: Filtering and Correction

```bash
cd data_preparation
python verify_responses.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

### Step 3: Training Data Preparation

```bash
cd data_preparation
python convert_format.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

```
## Inference
```
### Step 1: Model Inference

```bash
cd evalution
python phi_4_inference.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

### Step 2: Postprocessing

```bash
cd evalution
python postprocess.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

## Evaluations

### Step 1: Detection 

```bash
cd evalution
python eval_detection.py \
--input_file {input_file_path} \
--output_file {output_file_path} \
```

### Step 2: Editing 

```bash
cd evalution
python eval_factscore.py \
--input_file {input_file_path} \
```

