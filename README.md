# Data Quality Impact on Healthcare ML: Oral Cancer Case Study 📖 🏥

## Project Overview 👀
This repository contains my analysis of how data quality issues affect machine learning model performance in healthcare applications, using oral cancer prediction as a case study.

## Problem Statement 🎯
While many healthcare ML models report high accuracy, data quality issues like leakage and unrealistic class distributions can create misleading performance metrics. This project demonstrates these effects using a problematic oral cancer dataset.

## Dataset 📊
The analysis uses a dataset of 84,922 samples with 25 features related to oral cancer risk factors, symptoms, and diagnosis information. Key issues include unrealistic 50/50 cancer distribution and post-diagnosis features.

## Methodology 👨‍💻
The implementation compares three experimental conditions using XGBoost:
1. With data leakage (all features)
2. Without data leakage (pre-diagnosis features only)
3. Realistic class weight simulation (matching global prevalence rates)

## Evaluation 🤔
Model performance is evaluated using accuracy, precision, recall, F1, and ROC-AUC scores. Feature importance analysis demonstrates how data quality issues affect model decision-making and interpretation.

## Goals 🥅
- Demonstrate the impact of data leakage on model performance
- Illustrate how unrealistic class distributions affect evaluation metrics
- Show how feature importance shifts based on data quality issues
- Emphasize the importance of domain knowledge in healthcare ML applications
