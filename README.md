# Predicting venous thromboembolism in cancer patients using contrastive learning

This repository contains the code generated during the Master's Thesis titled: "Predicting venous thromboembolism in cancer patients using contrastive learning", developed in partial fulfillment of the requirements for the degree of MSc in Fundamental Principles of Data Science of the University of Barcelona.

+ Author: Júlia Villaró Cañizal
+ Program: MSc in Fundamental Principles of Data Science
+ Institution: University of Barcelona
+ Advisor: Dr. Oriol Pujol Vila and Bárbara Lobato Delgado

## Abstract

This thesis studies whether contrastive learning can improve the prediction of venous thromboembolism (VTE) in cancer patients using a particular dataset, which contains very few samples and a large number of clinical variables. In this setting, standard supervised learning models often struggle to capture the complex patterns required for accurate prediction. To address these limitations, the work investigates whether contrastive learning can learn more robust patient representations from limited data. 

It's also examined whether adding a feature gating mechanism to the contrastive model can down‑weight less informative variables, which would improve model interpretability.

## Repository

This repository contains all materials related to the thesis, serving as a comprehensive guide. It includes:
+ $\texttt{data}$ folder: Contains the dataset provided for the project. Due to patient confidentiality, the raw dataset cannot be uploaded, only the description of each clinical variable
+ $\texttt{dataframes}$ folder: Contains CSV files generated during preprocessing and the synthetic dataset created to mimic the statistical behaviour of the real cohort. For confidentiality reasons, preprocessed real-data CSVs are not included, only the features' traits.
+ $\texttt{images}$ folder: Contains plots and visualizations generated during exploratory analysis and model evaluation, such as correlation matrices, ROC and PR curves.
+ $\texttt{results}$ folder: Contains CSV files with the ROC‑AUC, PR‑AUC, and accuracy metrics for each model. Each file corresponds to a specific notebook and experiment.
+ Notebooks: The notebooks follow the full workflow of the thesis.
  + $\texttt{0.synthetic.ipynb}$ — pipeline to generate the synthetic dataset from the real one
  + $\texttt{1.ipynb}$, $\texttt{2.ipynb}$ and $\texttt{3.ipynb}$ — preprocessing of the real dataset, transforming it into the final modeling input
  + $\texttt{4.ipynb}$, $\texttt{5.ipynb}$, $\texttt{6.ipynb}$, $\texttt{7.ipynb}$, $\texttt{8.ipynb}$ — model training notebooks (contrastive, gating, baselines, etc.)
  + $\texttt{4.Synthetic.ipynb}$, $\texttt{5.Synthetic.ipynb}$, $\texttt{6.Synthetic.ipynb}$, $\texttt{7.Synthetic.ipynb}$, $\texttt{8.Synthetic.ipynb}$ — same models trained on the synthetic dataset for reproducibility
  + $\texttt{9.Results.ipynb}$ — code to generate all figures (ROC curves, PR curves, comparison plots, etc.)
+ $\texttt{utils.py}$: Contains helper functions used across notebooks to keep them clean and modular (data loading, preprocessing utilities, plotting helpers, etc.).

## Reproductibility

All code required to reproduce the experiments is included. Due to confidentiality restrictions, the raw real dataset is not provided, but:
+ The synthetic dataset allows full reproduction of the pipeline
+ All preprocessing steps are documented in the notebooks
+ All its model configurations and results are included

## Contact

Feel free to contact me to discuss any issues, questions or comments.
+ Email: juliavillaro02@gmail.com
+ Github: jvc02
