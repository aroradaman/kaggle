import os

import dabl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# load train and test
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# drop id and combine train+test
train_id = train["Id"]
test_id = test["Id"]
train = train.drop(["Id"], axis=1)

# analyze feature types
types = dabl.detect_types(train)
continuous_features = types[types["continuous"]].index.values.tolist()
useless_features = types[types["useless"]].index.values.tolist()
categorical_features = types[types["categorical"]].index.values.tolist()
categorical_features += types[types["low_card_int"]].index.values.tolist()

# visualizations
if not os.path.exists("plots"):
    os.mkdir("plots")

# box plots for categorical features
for var in categorical_features:
    fn = "plots/box_{}.png".format(var.lower())
    f, ax = plt.subplots(figsize=(10, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=train[["SalePrice", var]])
    plt.savefig(fn)
    print("{} Saved!".format(fn))
    plt.close()

# pair-plot for continuous variables
# to analyze distribution and correlation
# check skewness too
for var in continuous_features:
    skewness = train[var].skew()
    if skewness > 1 or skewness < -1:
        try:
            fn = "plots/dist_{}.png".format(var.lower())
            sns.distplot(train[var], fit=stats.norm);
            plt.savefig(fn)
            print("{} Saved!".format(fn))

            fn = "plots/prob_{}.png".format(var.lower())
            fig = plt.figure()
            res = stats.probplot(train[var], plot=plt)
            plt.savefig(fn)
            print("{} Saved!".format(fn))
        except RuntimeError:
            pass

    plt.savefig(fn)
    print("{} Saved!".format(fn))
    if var == "SalePrice":
        continue
    fn = "plots/reg_{}.png".format(var.lower())
    f, ax = plt.subplots(figsize=(10, 8))
    fig = sns.regplot(x=var, y="SalePrice", data=train[["SalePrice", var]])
    plt.savefig(fn)
    print("{} Saved!".format(fn))
    plt.close()

# correlation matrix
# understanding correlation wrt. sale price
# top 10 vars
k = 10
cols = train.corr().nlargest(k, "SalePrice")["SalePrice"].index
cm = np.corrcoef(train[cols].values.T)
influential_cols = cols.values
sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
    yticklabels=influential_cols, xticklabels=influential_cols
)
fn = "plots/corr_matrix.png"
plt.savefig(fn)
print("{} Saved!".format(fn))

sns.pairplot(train[influential_cols], size=2.5, kind="reg")
sns.set()
fn = "plots/pair_plot.png"
plt.savefig(fn)
print("{} Saved!".format(fn))
