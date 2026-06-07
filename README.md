# MasterFinalThesis
MSc Final Thesis - Predicting venous thromboembolism in cancer patients using contrastive learning

\begin{itemize}
    \item BBDD folder.
    
    \item ANALYSIS folder: 
    \newline 0.Getting started, 
    \newline 1.VTE\_vs\_BloodTest, 
    \newline 2.Barbara\_Dataset\_Analysis
    \newline Correlation matrix, variance computations
    
    \item MODEL folder: 
    \newline 1.0.Basic\_Model\_Data: treat the data for modelling, drop variables, one-hot-encoding, etc. Extract the correlation matrix
    \newline 1.0.Basic\_Model\_Nans: treat the data for modelling, fill in nan values
    \newline 1.1.Basic\_Models: try basic models (random forest, logistic regression, gradient boosting, neural network)
    \newline 1.1.Basic\_Models: try basic models with reduced stratified training set (random forest, logistic regression, gradient boosting, neural network) 
    \newline 2.0Contrastive\_Model\_baseline: trying an unsupervised method. 
    \newline 2.1.Contrastive\_Model\_v2: better baseline, proposed model with the best loss and product found. Also, when dropping manually some variables, the accuracy increases
    \newline 2.1.Contrastive\_Model\_v2\_Strat: same as before, but with reduced stratified training set.
    \newline 3.1.GATING from 2.1\_v2\_best: applying a gating, that assigns some weights to the variables and then minimizes the CL and a $\lambda$$L_1$$|w|$. 
\end{itemize}
