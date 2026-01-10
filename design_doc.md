#  Math Question Classification using Machine Learning and Gemini LLM

##  Project Overview

This project focuses on **automatically classifying math questions** into subtopics such as:

- Algebra  
- Geometry  
- Counting & Probability  
- Number Theory  
- Prealgebra  
- Intermediate Algebra  
- Precalculus  

Each question is stored as a JSON file containing a math problem, its level, and solution.  
The system uses **classical machine learning (TF-IDF + Logistic Regression/SVM)** for classification, and integrates **Google Gemini 2.5 Pro** to generate **step-by-step, student-friendly explanations** for each math problem.

##  Objective

- Classify math questions into correct subtopics using textual and symbolic features.  
- Design a math-aware text preprocessing pipeline that handles LaTeX and special symbols.  
- Train and evaluate a baseline ML model using TF-IDF features.  
- Integrate **LLM-based explanations** using Gemini 2.5 Pro.  
- Track all experiments and metrics using **Weights & Biases (wandb)**.  

##  Project Structure

```
dataset/
│
├── train_and_evaluate.py              Main ML pipeline
├── train_and_evaluate_ablation.py     To run multiple ablation experiments
├── design_doc.md                      For your report / documentation
│
├── src/                               All source scripts
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── explain_llm.py
│
├── wandb/                             Experiment tracking (auto-created)
├── MATH/                              Your dataset (train/test JSONs)
├── venv/                              Virtual environment
├── requirements.txt                   Dependencies list
└── .vscode/                           VS Code project settings

```

##  Workflow Overview

1️. **Data Loading** – loads JSON files into Pandas DataFrame.  
2️. **Preprocessing** – cleans LaTeX, normalizes math symbols, preserves equations.  
3️. **TF-IDF Vectorization** – converts text into numerical features.  
4️. **Model Training** – trains Logistic Regression or SVM classifier.  
5️. **Evaluation** – computes accuracy and confusion matrix.  
6️. **Experiment Tracking** – logs runs with Weights & Biases.  
7️. **LLM Explanation** – uses Gemini 2.5 Flash to generate explanations.

##  Libraries Used

| Library | Purpose |
|----------|----------|
| pandas | Data handling and DataFrame management |
| json, os, tqdm | File I/O and progress tracking |
| re, unicodedata | Text cleaning and normalization |
| scikit-learn | TF-IDF, model training, evaluation |
| wandb | Experiment tracking |
| google-genai | Gemini 2.5 Pro integration |
| matplotlib | Visualization |

##  Results 

| **Metric**            | **Value**  |
| --------------------- | ---------- |
| Accuracy              | **0.7254** |
| Macro Avg Precision   | 0.73       |
| Macro Avg Recall      | 0.75       |
| Macro Avg F1-Score    | 0.73       |
| Weighted Avg F1-Score | 0.72       |

## Ablation Study

| **Metric**            | **Value**  |
| --------------------- | ---------- |
| Accuracy              | **0.7254** |
| Macro Avg Precision   | 0.73       |
| Macro Avg Recall      | 0.75       |
| Macro Avg F1-Score    | 0.73       |
| Weighted Avg F1-Score | 0.72       |




## 🧰 Setup Instructions

```bash
git clone <your-repo-url>
cd dataset
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
python train_and_evaluate.py
```

## 🏁 Conclusion

This project demonstrates how **math-aware text preprocessing** and **classical ML techniques** can yield strong results on structured symbolic data.  
With **TF-IDF**, **Logistic Regression/SVM**, and **Gemini-based explanations**, the system achieves high accuracy while providing human-readable reasoning.

**Author:** Namish Garimella
**Date:** January 2026  
**Tools:** Python · Scikit-learn · W&B · Google Gemini 2.5 Flash
