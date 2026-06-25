import warnings
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency

import torch
from torch.utils.data import Dataset
import torch.nn as nn

from tqdm import tqdm



"""
-------------------------------------------------------------------------------------------------------------
Utils for notebook 2. 
-------------------------------------------------------------------------------------------------------------
"""


# Prints a frequency table for VTE and two other variables
def freq_table_pair(df, var1, var2):
    tab = (
        df.groupby("VTE")[[var1, var2]]
          .value_counts(dropna=False)
          .rename("Freq")
          .to_frame()
    )

    # Add % inside each VTE group
    tab["%"] = (
        tab["Freq"] /
        tab.groupby(level=0)["Freq"].transform("sum") * 100
    ).round(2)

    return tab


# Shannon entropy for categorical features
def categorical_entropy(series):
    counts = series.value_counts(dropna=False)
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()


# Frequency table for all VTE with all variables
def freq_table(df, columnas):
    tablas = {}

    for col in columnas:
        tabla = (
            df.groupby("VTE")[col]
              .value_counts(dropna=False)
              .rename("Freq")
              .to_frame()
        )

        # Adding % inside every VTE group
        tabla["%"] = (
            tabla["Freq"] /
            tabla.groupby(level=0)["Freq"].transform("sum") * 100
        ).round(2)

        tablas[col] = tabla

    return tablas


# Score: sum of absolute differences between category percentages in VTE=1 and VTE=0. Returns a single numeric score
def association_score(tab):
    # Pivot: rows = category, columns = VTE, values = %
    pivot = tab["%"].unstack(level=0).fillna(0)

    # If only one VTE group exists, then no association
    if pivot.shape[1] < 2:
        return 0

    # Sum of absolute differences between VTE groups
    score = np.abs(pivot.iloc[:,0] - pivot.iloc[:,1]).sum()
    return score


# Prints a table with three columns: column | type | distinct values
def table_types_unique(df):
    rows = []

    for col in df.columns:
        serie = df[col]

        # Type
        if pd.api.types.is_numeric_dtype(serie):
            tipo = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(serie):
            tipo = "datetime"
        elif pd.api.types.is_bool_dtype(serie):
            tipo = "boolean"
        else:
            tipo = "categorical"

        rows.append({
            "column": col,
            "type": tipo,
            "distinct_values": serie.nunique(dropna=True)
        })

    return pd.DataFrame(rows)


# Computes a Pearson correlation matrix only for numerical variables
def numeric_corr_matrix(df):
    # Only numerical columns
    numeric_cols = df.select_dtypes(include=["number"]).columns

    # Pearson matrix
    corr = df[numeric_cols].corr(method="pearson").abs()

    return corr


# Computes Cramér’s V (bias‑corrected), a measure of association between two categorical variables
# 0 = no relation, 1 = perfect relation
def cramers_v(x, y): 
    # Drop rows where x or y are NaN 
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if valid.empty:
        return np.nan
    
    confusion_matrix = pd.crosstab(valid["x"], valid["y"])
    
    # If the table is too small -> undefined correlation
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return np.nan
    
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return np.nan
    
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # Bias correction
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    denom = min((kcorr-1), (rcorr-1))
    if denom <= 0:
        return np.nan
    
    return np.sqrt(phi2corr / denom)


# Computes the Correlation Ratio, which measures how strongly a categorical variable explains the variance of a numerical variable
def correlation_ratio(categories, values):
    # Delete NaNs in both variables
    valid = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if valid.empty:
        return np.nan

    categories = valid["cat"]
    values = valid["val"]

    # Mean values by category
    cat_groups = values.groupby(categories)
    means = cat_groups.mean()
    counts = cat_groups.count()

    # Global mean
    overall_mean = values.mean()

    # Squares sum between groups
    ss_between = np.sum(counts * (means - overall_mean)**2)

    # Total squared sum
    ss_total = np.sum((values - overall_mean)**2)

    if ss_total == 0:
        return np.nan

    eta = np.sqrt(ss_between / ss_total)
    return eta


# Builds a full mixed‑type correlation (square) matrix for a dataframe containing numerical and categorical variables
# For each pair of variables:
# - num–num -> Pearson
# - cat–cat -> Cramér’s V
# - num–cat or cat-num -> Correlation Ratio
def mixed_corr_matrix(df):
    cols = df.columns
    n = len(cols)

    corr = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    # Identify types
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Obtain numerical matrix
    corr_num = numeric_corr_matrix(df)

    for i in range(n):
        for j in range(n):
            col_i = cols[i]
            col_j = cols[j]

            # num - num -> use numeric_corr_matrix
            if col_i in num_cols and col_j in num_cols:
                corr.iloc[i, j] = corr_num.loc[col_i, col_j]

            # cat - cat -> use Cramér’s V
            elif col_i in cat_cols and col_j in cat_cols:
                corr.iloc[i, j] = cramers_v(df[col_i], df[col_j])

            # num - cat -> Correlation Ratio
            elif col_i in num_cols and col_j in cat_cols:
                corr.iloc[i, j] = correlation_ratio(df[col_j], df[col_i])

            # cat - num -> Correlation Ratio
            elif col_i in cat_cols and col_j in num_cols:
                corr.iloc[i, j] = correlation_ratio(df[col_i], df[col_j])

    return corr



"""
-------------------------------------------------------------------------------------------------------------
Utils for notebook 3. 
-------------------------------------------------------------------------------------------------------------
"""


# Returns a table with three columns: column | distinct values | nan
def num_nans(df):
    rows = []

    for col in df.columns:
        serie = df[col]

        rows.append({
            "column": col,
            "distinct_values": serie.nunique(dropna=True),
            "nan": serie.isna().sum()
        })

    return pd.DataFrame(rows)


# Imputes missing values in a categorical column by sampling from the observed distribution of that column
def impute_with_distribution(df, column):

    # Distribution of observed values
    value_counts = df[column].value_counts(normalize=True)
    categories = value_counts.index.values
    probabilities = value_counts.values

    # Mask of missing values
    mask = df[column].isna()

    # Sample according to empirical distribution
    df.loc[mask, column] = np.random.choice(
        categories,
        size=mask.sum(),
        p=probabilities
    )

    # Optional: print remaining NaNs
    n_missing = df[column].isna().sum()
    print(f"Number of NaN values in {column}: {n_missing}")

    return df


# Impute missing values for two correlated categorical variables: pTNM_stage (categorical) and tumor_surgically_removed (binary 0/1)
# Missing values are filled using:
# - P(removed | stage) when stage is known
# - P(stage | removed) when removed is known
# - P(stage, removed) when both are missing
def impute_stage_and_surgery(df, col_stage, col_removed):
    
    # 1. Build joint distribution P(stage, removed)
    joint = (
        df.dropna(subset=[col_stage, col_removed])
        .groupby([col_stage, col_removed])
        .size()
        .reset_index(name="count")
    )

    joint["prob"] = joint["count"] / joint["count"].sum()

    # 2. Conditional P(removed | stage)
    cond_removed_given_stage = joint.copy()
    cond_removed_given_stage["prob"] = (
        cond_removed_given_stage.groupby(col_stage)["count"]
        .transform(lambda x: x / x.sum())
    )

    # 3. Conditional P(stage | removed)
    cond_stage_given_removed = joint.copy()
    cond_stage_given_removed["prob"] = (
        cond_stage_given_removed.groupby(col_removed)["count"]
        .transform(lambda x: x / x.sum())
    )

    # 4. Row-wise imputation logic
    def _impute_row(row):
        stage = row[col_stage]
        removed = row[col_removed]

        # Case 1 — nothing missing
        if pd.notna(stage) and pd.notna(removed):
            return stage, removed

        # Case 2 — stage known, removed missing
        if pd.notna(stage) and pd.isna(removed):
            subset = cond_removed_given_stage[
                cond_removed_given_stage[col_stage] == stage
            ]
            removed_new = np.random.choice(subset[col_removed], p=subset["prob"])
            return stage, removed_new

        # Case 3 — removed known, stage missing
        if pd.isna(stage) and pd.notna(removed):
            subset = cond_stage_given_removed[
                cond_stage_given_removed[col_removed] == removed
            ]
            stage_new = np.random.choice(subset[col_stage], p=subset["prob"])
            return stage_new, removed

        # Case 4 — both missing -> sample from joint P(stage, removed)
        choice = np.random.choice(joint.index, p=joint["prob"])
        row_choice = joint.loc[choice]
        return row_choice[col_stage], row_choice[col_removed]

    # 5. Apply imputation
    for idx, row in df.iterrows():
        new_stage, new_removed = _impute_row(row)
        df.loc[idx, col_stage] = new_stage
        df.loc[idx, col_removed] = new_removed

    return df


# Impute missing values for two correlated categorical variables: primary tumor (one-hot encoded, exactly one 1) and catheter_device (binary 0/1)
# Missing values are filled using:
# - If tumor is known and catheter is missing then sample P(catheter | tumor)
# - If catheter is known and tumor is missing then sample P(tumor | catheter)
# - If both missing then sample from joint P(tumor, catheter)
# - If nothing missing then leave unchanged
def impute_primary_and_catheter(df, primary_cols, catheter_col):
 
    # 1. Build joint counts for all (tumor, catheter) pairs
    pairs = []
    for tumor in primary_cols:
        for cat in [0, 1]:
            count = len(df[(df[tumor] == 1) & (df[catheter_col] == cat)])
            pairs.append([tumor, cat, count])

    joint_df = pd.DataFrame(pairs, columns=["primary_tumor", "catheter_device", "count"])

    # Joint probability P(tumor, catheter)
    joint_df["prob"] = joint_df["count"] / joint_df["count"].sum()

    # 2. Conditional probabilities
    # P(catheter | tumor)
    cond_cat_given_tumor = joint_df.copy()
    cond_cat_given_tumor["prob"] = (
        cond_cat_given_tumor.groupby("primary_tumor")["count"]
        .transform(lambda x: x / x.sum())
    )

    # P(tumor | catheter)
    cond_tumor_given_cat = joint_df.copy()
    cond_tumor_given_cat["prob"] = (
        cond_tumor_given_cat.groupby("catheter_device")["count"]
        .transform(lambda x: x / x.sum())
    )

    # 3. Row-wise imputation logic
    def _impute_row(row):
        catheter = row[catheter_col]
        tumor_values = row[primary_cols].values

        has_tumor = tumor_values.sum() == 1
        all_zero_tumor = tumor_values.sum() == 0

        # Case 1 — nothing missing
        if pd.notna(catheter) and has_tumor:
            return row[primary_cols].values, catheter

        # Case 2 — catheter missing, tumor known
        if pd.isna(catheter) and has_tumor:
            tumor_col = primary_cols[np.argmax(tumor_values)]
            subset = cond_cat_given_tumor[
                cond_cat_given_tumor["primary_tumor"] == tumor_col
            ]
            catheter_new = np.random.choice(subset["catheter_device"], p=subset["prob"])
            return row[primary_cols].values, catheter_new

        # Case 3 — tumor missing, catheter known
        if all_zero_tumor and pd.notna(catheter):
            subset = cond_tumor_given_cat[
                cond_tumor_given_cat["catheter_device"] == catheter
            ]
            chosen_tumor = np.random.choice(subset["primary_tumor"], p=subset["prob"])
            tumor_vector = np.zeros(len(primary_cols))
            tumor_vector[primary_cols.index(chosen_tumor)] = 1
            return tumor_vector, catheter

        # Case 4 — both missing → sample from joint distribution
        choice = np.random.choice(joint_df.index, p=joint_df["prob"])
        row_choice = joint_df.loc[choice]

        tumor_vector = np.zeros(len(primary_cols))
        tumor_vector[primary_cols.index(row_choice["primary_tumor"])] = 1

        return tumor_vector, row_choice["catheter_device"]

    # 4. Apply imputation
    for idx, row in df.iterrows():
        tumor_vec, catheter_val = _impute_row(row)
        df.loc[idx, primary_cols] = tumor_vec
        df.loc[idx, catheter_col] = catheter_val

    return df


# Impute missing values in a one-hot encoded categorical variable
# A row is considered missing if all one-hot columns are 0
# Imputation is done by sampling one category according to the empirical distribution of 1s across the one-hot columns
def impute_onehot_with_distribution(df, onehot_cols):

    # Count how many 1s each category has (empirical distribution)
    counts = df[onehot_cols].sum()
    total = counts.sum()

    if total == 0:
        raise ValueError("All one-hot columns are zero; cannot compute distribution.")

    probabilities = counts / total

    # Identify rows where all one-hot columns are 0 → missing category
    mask_missing = (df[onehot_cols].sum(axis=1) == 0)
    rows_to_fix = df[mask_missing].index

    categories = np.array(onehot_cols)
    probs = probabilities.values

    # Impute: set exactly one column to 1
    for idx in rows_to_fix:
        chosen_col = np.random.choice(categories, p=probs)
        df.loc[idx, chosen_col] = 1

    return df



"""
-------------------------------------------------------------------------------------------------------------
Utils for notebook 5. 
-------------------------------------------------------------------------------------------------------------
"""


# This class converts NumPy arrays into a PyTorch‑compatible dataset that can be indexed and batched automatically during training.
# The constructor receives the feature matrix (X) and target vector (y), converting them into PyTorch tensors
# During training, the DataLoader repeatedly calls this method to construct mini-batches. 
# Consequently, the DataLoader can automatically generate batches of size 16 and shuffle the training data at each epoch.
class TabularDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):              # returns the total number of samples, allowing PyTorch to determine dataset size
        return len(self.X)

    def __getitem__(self, idx):     # retrieves a single sample-label pair given an index
        return self.X[idx], self.y[idx]


# The representation learner: transforms the original feature vector into a 16‑dimensional embedding that captures VTE‑related structure.
# It doesn't classify. It only produces embeddings that contain useful information for distinguishing between VTE and non-VTE patients.
class Encoder(nn.Module):

    def __init__(self, d_init, d_out):
        super().__init__()
        # self.block1 = nn.Sequential(
        #     nn.Linear(d_init, 64),
        #     nn.LayerNorm(64),
        #     nn.LeakyReLU(0.1),
        #     nn.Dropout(0.2)
        # )
        self.block2 = nn.Sequential(
            nn.Linear(d_init, 32),
            nn.LayerNorm(32),        # stabilizes training by ensuring that activations remain within a controlled range
            nn.LeakyReLU(0.1)#,      # LeakyReLU allows a small gradient for negative values, reducing the risk of inactive neurons
            #nn.Dropout(0.2)
        )
        self.head = nn.Linear(32, d_out)  # This produces the embedding vector.

    def forward(self, x):
        # x = self.block1(x)
        x = self.block2(x)
        emb = self.head(x)  # 16-d embedding
        return emb


# Supervised Training of the Encoder
# This class wraps the encoder and adds a temporary linear classifier head (to provide a supervised learning signal)
# because the encoder must be trained supervised, using the VTE label, so it learns embeddings that separate classes.
# The neural network learns through the NNClassifier, and the encoder kepts is shaped by that learning
class NNClassifier(nn.Module):

    def __init__(self, d_init, d_emb):
        super().__init__()
        self.encoder = Encoder(d_init=d_init, d_out=d_emb)
        self.classifier_head = nn.Linear(d_emb, 1)  # single logit for binary classification

    def forward(self, x):
        emb = self.encoder(x)
        logit = self.classifier_head(emb)
        return logit, emb


# It performs one full training pass over the dataset
# A batch of samples is loaded from the DataLoader.
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0   # Will store the sum of losses across all batches

    for X_batch, y_batch in loader:
        # Move data to GPU/CPU
        # Inputs and labels are transferred to the selected device (CPU or GPU).
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float().view(-1, 1)  # shape (batch_size, 1)

        # Reset gradients
        optimizer.zero_grad()
        # Forward pass, the batch is propagated through the network.
        logits, _ = model(X_batch)
        # Compute loss
        loss = criterion(logits, y_batch)
        # Backpropagation calculates gradients.
        loss.backward()
        # Update model weights via AdamW
        optimizer.step()

        # Accumulate batch loss
        total_loss += loss.item() * X_batch.size(0)

    # Return average loss
    return total_loss / len(loader.dataset)


# Evaluates the model on validation or test data
# Model performance during training is monitored by this function by following the same forward-pass 
# procedure as training but disables gradient computation and parameter updates:
# - Inputs are passed through the network.
# - Predictions are generated.
# - The loss is computed.
# - Loss values are accumulated.
def evaluate_loss(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().view(-1, 1)

            logits, _ = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)

    # average loss, which gives an estimate of the model’s generalization performance and can help detect overfitting
    return total_loss / len(loader.dataset)  


# Transforms the entire dataset into embeddings using the trained encoder.
# For each batch, it extracts the latent representation (emb) and stores it,
# returning a compact embedding matrix Xt and the corresponding labels yt.
def transform_dataset(model, loader, device):
    
    Z_list = []         # store the embeddings (outputs of model.forward_once) for all samples
    y_list = []         # store the corresponding labels

    # evaluation mode (no dropout, no batchnorm updates)
    model.eval()
    
    # don’t track gradients inside this block (not needed)
    with torch.no_grad():
       
        # Iterates over the original dataset in batches
        for Xb, y in loader:
            Xb, y = Xb.to(device), y.to(device)     # Xb = batch of input features (no pairs, just single samples)
            
            # The model returns (logit, emb). We only want emb [1]
            _, emb = model.forward(Xb) 
            
            # Apply .cpu() to the extracted embedding tensor
            Z_list.append(emb.cpu())
            y_list.append(y.cpu())

    Xt = torch.cat(Z_list, dim=0).numpy()
    yt = torch.cat(y_list, dim=0).numpy()

    # Xt is the full embedding matrix: one 16‑dimensional vector per patient.
    # yt contains the corresponding labels.
    # These embeddings replace the original features and are used as input to the downstream classifier.
    return Xt, yt



"""
-------------------------------------------------------------------------------------------------------------
Utils for notebooks 6. and 8.
-------------------------------------------------------------------------------------------------------------
"""


# Dataset classes: This is the core of the contrastive setup

# For any given sample, it randomly selects another sample from the dataset, creating a pair 
# It uses a pos_fraction parameter to ensure a controlled mix of positive pairs (same class) and negative pairs (different class).
# This creates more training signals by comparing samples rather than just looking at them individually. 
class PairDatasetBalanced(Dataset):
    def __init__(self, X, y, pos_fraction=0.5):   # probability of generating a positive pair
        self.X = X
        self.y = y
        self.pos_fraction = float(pos_fraction)

        # Convert labels to a clean NumPy array
        # it ensures labels are 1D, labels are integers and they can be indexed easily
        if torch.is_tensor(y):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)
            
        y_np = np.asarray(y_np).reshape(-1)       # ensures labels are a 1D vector
        y_np = y_np.astype(np.int64, copy=False)  # ensures labels are integer class IDs

        # Store basic info
        self.y_np = y_np
        self.n = len(y_np)

        # list of indices, crucial for sampling positive pairs
        # class 0 -> [0, 3, 5, 10, ...]
        # class 1 -> [1, 2, 4, 7, ...]
        self.class_to_idx = {}
        for i, c in enumerate(y_np):
            self.class_to_idx.setdefault(int(c), []).append(i)

        self.classes = list(self.class_to_idx.keys())

        # indices of all other classes, used for negative pairs
        # not_class_idx[0] -> all indices where label != 0
        # not_class_idx[1] -> all indices where label != 1
        all_idx = np.arange(self.n)
        self.not_class_idx = {}
        for c in self.classes:
            mask = (y_np != c)
            self.not_class_idx[c] = all_idx[mask]
            
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # pick the anchor sample
        xa = self.X[idx]
        ya = self.y[idx]
        c = int(self.y_np[idx])

        # decide whether to make a positive or negative pair:
        # With probability pos_fraction, try to make a positive pair, but only if the class has at least 2 samples
        make_pos = (np.random.rand() < self.pos_fraction) and (len(self.class_to_idx[c]) > 1)

        if make_pos:
            # sample the second element of the pair: pick another sample from the same class and ensure it is not the same index
            pos_pool = self.class_to_idx[c]
            j = idx
            while j == idx:
                j = pos_pool[np.random.randint(len(pos_pool))]
        else:          # if negative pair, pick a sample from a different class
            neg_pool = self.not_class_idx[c]
            j = int(neg_pool[np.random.randint(len(neg_pool))])

        # return the pair
        xb = self.X[j]
        yb = self.y[j]
        return xa, xb, ya, yb      # anchor sample, paired sample, anchor label, pair label


# Model and loss
# Neural Network with 1 hidden layer and a LeakyReLU nonlinearity
class DeepKernel(nn.Module):
    def __init__(self, d_init, d_out):
        super().__init__()
        #self.block1 = nn.Sequential(
            #nn.Linear(d_init, 64),
            #nn.LayerNorm(64),
            #nn.LeakyReLU(0.1),
            #nn.Dropout(0.2)
        #)
        self.block2 = nn.Sequential(
            nn.Linear(d_init, 32),  # 64
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1)#,         nonlinear activation
        #     nn.Dropout(0.2)
        )
        self.head = nn.Linear(32, d_out)

    def forward_once(self, x):
        #x = self.block1(x)
        x = self.block2(x)
        return self.head(x)

    # compute dot-product similarity between two samples
    def forward(self, xa, xb):
        s1 = self.forward_once(xa)
        s2 = self.forward_once(xb)
        return torch.sum(s1 * s2, dim=1)    # <- dot-product similarity


# If embeddings are unit vectors, then the dot product between two embeddings equals their cosine similarity.
# This is not used. We'll use hinge_loss
def cosine_hinge_loss(similarity, target, margin):
    pos = target * (1 - similarity)     # target=1
    neg = (1 - target) * torch.clamp(similarity - margin, min=0)
    return torch.mean(pos + neg)


# Penalizes positive pairs if they aren't similar enough and negative pairs if they are too similar
def hinge_loss(similarity, target, margin):       
    # the margin controls how far apart negative pairs should be, so after this margin, loss will be for negative pairs
    pos = target * torch.clamp(1 - similarity, min=0)
    neg = (1 - target) * torch.clamp(similarity - margin, min=0)
    return torch.mean(pos + neg)


# Training and evaluation loops
# - calculates the dot-product similarity between pairs
# - computes the loss against the target (whether the pair is actually the same class)
# - updates weights
def train_one_epoch_contrastive(dataloader, model, loss_fn, optimizer, device, margin):
    # Enables dropout, gradient updates, etc
    model.train()
    
    running_loss = 0.0   # accumulates the loss across the epoch
    pbar = tqdm(dataloader, desc="train", leave=False)   # creates a progress bar for the batches (it shows progress visually)
    
    for Xa, Xb, ya, yb in pbar:
        Xa, Xb = Xa.to(device), Xb.to(device)
        ya, yb = ya.to(device), yb.to(device)

        # build the target vector (produces 1 por positive pairs and 0 for negative pairs)
        target = (ya == yb).float().view(-1)
        
        optimizer.zero_grad()   # clears the gradients before computing new ones

        # returns cosine similarity between embeddings. It's the same as dot-product similarity (return torch.sum(s1 * s2, dim=1))
        pred = model(Xa, Xb).view(-1)
        
        loss = loss_fn(pred, target, margin=margin)

        # backward pass (updates the model weights)
        loss.backward()
        optimizer.step()

        # accumulate loss
        running_loss += loss.item()
        pbar.set_postfix(avg_loss=running_loss / (pbar.n + 1))   # updates the progress bar with the current average loss

    # Return average loss
    return running_loss / len(dataloader)


# Tracks loss and accuracy on a validation set without updating the model.
@torch.no_grad()
def eval_one_epoch(dataloader, model, loss_fn, device, margin, threshold):      # evaluation without training
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="eval", leave=False)
    for Xa, Xb, ya, yb in pbar:
        Xa, Xb = Xa.to(device), Xb.to(device)
        ya, yb = ya.to(device), yb.to(device)

        target = (ya == yb).float().view(-1)
        pred = model(Xa, Xb).view(-1)

        loss = loss_fn(pred, target, margin=margin)
        running_loss += loss.item()

        pred_label = (pred >= threshold).float()
        correct += (pred_label == target).sum().item()
        total += target.numel()

        pbar.set_postfix(
            avg_loss=running_loss / (pbar.n + 1),
            acc=correct / max(total, 1)
        )

    # Return loss and accuracy
    return running_loss / len(dataloader), correct / max(total, 1)


# Fit the model
# - calls the training/eval loops
# - manages a learning rate scheduler (to decrease the LR if the loss plateaus)
# - saves the history of loss and accuracy for plotting
def fit(train_loader, val_loader, model, loss_fn, optimizer, device,
        epochs, margin, threshold, scheduler, verbose=True):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(1, epochs + 1):
        # train
        train_loss = train_one_epoch_contrastive(train_loader, model, loss_fn, optimizer, device, margin=margin)
        # Validate
        val_loss, val_acc = eval_one_epoch(val_loader, model, loss_fn, device, margin=margin, threshold=threshold)

        # Scheduler step
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        # Print progress (if verbose = True)
        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"val_acc={val_acc*100:.2f}% | lr={lr:.2e}"
            )

    return history


# To visualize the training process so there's a check for convergence (where loss stops decreasing) 
# and ensure the model isn't diverging or overfitting (where validation loss starts increasing while training loss decreases)
def plot_history(history, LRshow=False):
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    plt.figure()
    plt.plot(list(epochs), history["train_loss"], label="train_loss")
    plt.plot(list(epochs), history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Convergence: Loss")
    plt.show()

    # Accuracy plot
    plt.figure()
    plt.plot(list(epochs), [100*a for a in history["val_acc"]], label="val_acc (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Convergence: Validation Accuracy")
    plt.show()

    # LR plot (optional but useful)
    if LRshow:
        plt.figure()
        plt.plot(list(epochs), history["lr"], label="lr")
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.legend()
        plt.title("Learning Rate Schedule")
        plt.show()


# Converts the entire dataset into embeddings using the trained contrastive encoder.
# For each batch, it extracts the embedding (forward_once) and stores it along with the labels.
def transform_dataset_contrastive(model, loader, device):
    
    Z_list = []      # store the embeddings (outputs of model.forward_once) for all samples
    y_list = []      # store the corresponding labels

    # Ensures embeddings are computed in a deterministic way (no dropout, no batchnorm updates)
    model.eval()     # evaluation mode
    
    # no gradients needed during inference
    with torch.no_grad():
        
        # Iterates over the original dataset in batches
        for Xb, y in loader:
            Xb, y = Xb.to(device), y.to(device)     # Xb = batch of input features (no pairs, just single samples)
            
            # forward_once returns the embedding vector for each sample
            # Output pred shape: [batch_size, d_out=16]
            pred = model.forward_once(Xb)
            
            # Appends them to the lists
            Z_list.append(pred.cpu())   # Z_list: list of tensors of shape [batch_size, d_out].
            y_list.append(y.cpu())      # y_list: list of tensors of shape [batch_size].

    Xt = torch.cat(Z_list, dim=0).numpy()
    yt = torch.cat(y_list, dim=0).numpy()

    # Xt = embedding matrix (new feature space), yt = aligned labels
    return Xt, yt



"""
-------------------------------------------------------------------------------------------------------------
Utils for notebook 7. 
-------------------------------------------------------------------------------------------------------------
"""


# GatingLayer: learns a weight for each input feature and applies it element‑wise.
# The parameter vector v (size = input_dim) is learned during training.
# In the forward pass, each feature x_i is multiplied by its learned weight v_i,
# allowing the model to amplify informative features and suppress less useful ones.
class GatingLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.v = nn.Parameter(torch.randn(input_dim))
        # self.v = nn.Parameter(torch.zeros(input_dim))
        self.v = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        w = self.v
        # w = w / (w.sum() + 1e-8)
        return x * w
    

# Model and loss
# Neural Network with a gating layer + 1 hidden layer and a LeakyReLU nonlinearity
class DeepKernel_gating(nn.Module):
    def __init__(self, d_init, d_out):
        super().__init__()
        
        self.gating = GatingLayer(d_init)    # gating layer
        
        #self.block1 = nn.Sequential(
            #nn.Linear(d_init, 64),
            #nn.BatchNorm1d(64),
            #nn.LeakyReLU(0.1),
            #nn.Dropout(0.2)
        #)
        self.block2 = nn.Sequential(
            nn.Linear(d_init, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1)#,
            #nn.Dropout(0.2)
        )
        self.head = nn.Linear(32, d_out)
    
    def forward_once(self, x):
        x = self.gating(x)
        #x = self.block1(x)
        x = self.block2(x)
        return self.head(x)

    # compute dot-product similarity between two samples
    def forward(self, xa, xb):
        s1 = self.forward_once(xa)
        s2 = self.forward_once(xb)
        return torch.sum(s1 * s2, dim=1)    # <- dot-product similarity


# Training and evaluation loops
# - calculates the dot-product similarity between pairs
# - computes the loss against the target (whether the pair is actually the same class)
# - applies an L1 penalty (lambda_l1 * torch.sum(torch.abs(w))) on the gating weights to encourage sparsity
# - updates weights
def train_one_epoch_gating(dataloader, model, loss_fn, optimizer, device, margin, lambda_l1):
    # Enables dropout, gradient updates, etc
    model.train()
    
    running_loss = 0.0       # accumulates the loss across the epoch
    pbar = tqdm(dataloader, desc="train", leave=False)
    
    for Xa, Xb, ya, yb in pbar:
        Xa, Xb = Xa.to(device), Xb.to(device)
        ya, yb = ya.to(device), yb.to(device)

        # build the target vector (produces 1 por positive pairs and 0 for negative pairs)
        target = (ya == yb).float().view(-1)
        
        optimizer.zero_grad()   # clears the gradients before computing new ones

        # returns cosine similarity between embeddings. It's the same as dot-product similarity (return torch.sum(s1 * s2, dim=1))
        pred = model(Xa, Xb).view(-1)
        
        loss_contrastive = loss_fn(pred, target, margin=margin)

        # Get gating weights
        w = model.gating.v
        # w = w / (w.sum() + 1e-8)
        
        loss_l1 = lambda_l1 * torch.sum(torch.abs(w))
        
        loss = loss_contrastive + loss_l1

        # backward pass (updates the model weights)
        loss.backward()
        optimizer.step()
        
        # accumulate loss
        running_loss += loss.item()
        pbar.set_postfix(avg_loss=running_loss / (pbar.n + 1))   # updates the progress bar with the current average loss

    # Return average loss
    return running_loss / len(dataloader)


# Fit the model
# - calls the training/eval loops
# - manages a learning rate scheduler (to decrease the LR if the loss plateaus)
# - saves the history of loss and accuracy for plotting
# - applies L1 regularization on the gating weights to encourage sparsity and feature selection
def fit_gating(train_loader, val_loader, model, loss_fn, optimizer, device,
        epochs, margin, lambda_l1, threshold, scheduler, verbose=True):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(1, epochs + 1):
        # train
        train_loss = train_one_epoch_gating(train_loader, model, loss_fn, optimizer, device, margin=margin, lambda_l1=lambda_l1)
        # Validate
        val_loss, val_acc = eval_one_epoch(val_loader, model, loss_fn, device, margin=margin, threshold=threshold)

        # Scheduler step
        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        # Print progress (if verbose = True)
        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"val_acc={val_acc*100:.2f}% | lr={lr:.2e}"
            )

    return history



"""
-------------------------------------------------------------------------------------------------------------
Notebook: Results_Analysis (plotting)
-------------------------------------------------------------------------------------------------------------
"""


# Plots a metric (ROC-AUC, PR-AUC, Accuracy) vs test_size for any number of models
def compare_models(summary, metric, title=None):

    fig = plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=summary,
        x="test_size",
        y=metric,
        hue="model",
        marker="o",
        linewidth=3,
        markersize=9
    )

    if title is None:
        plt.title(f"{metric} vs Test Size – Model Comparison", fontsize=20)
    else:
        plt.title(title, fontsize=20)

    plt.xlabel("Test Size", fontsize=16)
    plt.ylabel(metric, fontsize=16)

    # percentage ticks
    test_sizes = sorted(summary["test_size"].unique())
    plt.xticks(
        test_sizes,
        [f"{int(ts*100)}%" for ts in test_sizes],
        fontsize=14,
        rotation=45
    )

    plt.yticks(fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14, title_fontsize=16)
    plt.tight_layout()

    return fig


# Plots a single model's metric (ROC-AUC, PR-AUC, Accuracy) vs test_size
def compare_test_size(summary, metric, title=None):

    model_name = summary["model"].iloc[0]

    fig = plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=summary,
        x="test_size",
        y=metric,
        marker="o",
        linewidth=3,
        markersize=9
    )

    # Title
    if title is None:
        plt.title(f"{metric} vs Test Size – {model_name}", fontsize=20)
    else:
        plt.title(title, fontsize=20)

    # Axis labels
    plt.xlabel("Test Size", fontsize=16)
    plt.ylabel(metric, fontsize=16)

    # Convert test_size to percentages
    test_sizes = sorted(summary["test_size"].unique())
    plt.xticks(
        test_sizes,
        [f"{int(ts*100)}%" for ts in test_sizes],
        fontsize=14,
        rotation=45
    )

    plt.yticks(fontsize=14)
    plt.grid(alpha=0.3)

    # Legend (top right)
    plt.legend(
        [model_name],
        fontsize=14,
        title_fontsize=16,
        loc="upper right"
    )

    plt.tight_layout()
    return fig


# Creates a twin-axis plot of ROC-AUC and PR-AUC vs test_size for a single model
def roc_pr_combined(summary):

    model_name = summary["model"].iloc[0]

    fig = plt.figure(figsize=(10, 6))

    # Left axis (ROC-AUC)
    ax1 = sns.lineplot(
        data=summary,
        x="test_size",
        y="ROC-AUC",
        marker="o",
        linewidth=2,
        markersize=8,
        color="tab:blue",
        label="ROC-AUC"
    )

    ax1.set_xlabel("Test Size", fontsize=16)
    ax1.set_ylabel("ROC-AUC", fontsize=16, color="tab:blue")
    ax1.tick_params(axis='y', labelsize=14, colors="tab:blue")
    ax1.grid(alpha=0.3)

    # Convert test_size to percentages
    test_sizes = sorted(summary["test_size"].unique())
    ax1.set_xticks(test_sizes)
    ax1.set_xticklabels([f"{int(ts*100)}%" for ts in test_sizes],
                        fontsize=14, rotation=45)

    # Right axis (PR-AUC)
    ax2 = ax1.twinx()

    sns.lineplot(
        data=summary,
        x="test_size",
        y="PR-AUC",
        marker="o",
        linewidth=2,
        markersize=8,
        color="tab:red",
        label="PR-AUC",
        ax=ax2
    )

    # Remove auto-legend from ax2
    ax2.legend_.remove()

    ax2.set_ylabel("PR-AUC", fontsize=16, color="tab:red")
    ax2.tick_params(axis='y', labelsize=14, colors="tab:red")

    # Title
    plt.title(f"ROC and PR vs Test Size – {model_name}", fontsize=20)

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2,
               labels_1 + labels_2,
               fontsize=14,
               title="Metric",
               title_fontsize=16,
               loc="upper right")

    plt.tight_layout()
    return fig
