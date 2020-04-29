import dabl
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="ticks", color_codes=True)
pd.set_option('display.max_rows', 500)


def printer(*args):
    print("_" * 150)
    for arg in args:
        print(arg)
        print()
    print("_" * 150)
    print()


# load train and test
train = pd.read_csv("data/train.csv")
printer("Training Data", train.shape)
test = pd.read_csv("data/test.csv")

# drop id and combine train+test
train_id = train["Id"]
test_id = test["Id"]
train = train.drop(["Id"], axis=1)
test = test.drop(["Id"], axis=1)
df = pd.concat([train, test])
df = df.drop(["SalePrice"], axis=1)
df = df.reset_index(drop=True)

# analyze feature types
types = dabl.detect_types(train)

continuous_features = types[types["continuous"]].index.values.tolist()
printer("Continuous Features", continuous_features)

categorical_features = types[types["categorical"]].index.values.tolist()
categorical_features += types[types["low_card_int"]].index.values.tolist()
printer("Categorical Features", continuous_features)

useless_features = types[types["useless"]].index.values.tolist()
printer("Useless Features", useless_features)

# finalize dtypes
for col in ["FullBath", "HalfBath"]:
    categorical_features.remove(col)
    continuous_features.append(col)
skewed_features = []

# correct type of categorical_features
for col in categorical_features:
    df[col] = df[col].astype(str)

# drop useless features
train = train.drop(types[types["useless"]].index.values, axis=1)
test = test.drop(types[types["useless"]].index.values, axis=1)
df = df.drop(types[types["useless"]].index.values, axis=1)

# missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=["total", "percent"])
missing_data["percent"] *= 100

missing_data = missing_data[missing_data["total"] > 0]
printer("Missing Data", missing_data)

# drop missing columns since they don't co relate much with target
train = train.drop((missing_data[missing_data["total"] > 1]).index, 1)
train = train.drop(train.loc[train["Electrical"].isnull()].index)

# analyzing dist plots and correcting skewness by log
train["SalePrice"] = np.log(train["SalePrice"])
train["GrLivArea"] = np.log(train["GrLivArea"])

# handling log 0 for columns with 0 vals
train["HasBsmt"] = pd.Series(len(train["TotalBsmtSF"]), index=train.index)
train["HasBsmt"] = 0
train.loc[train["TotalBsmtSF"] > 0, "HasBsmt"] = 1

# transform data
train.loc[train["HasBsmt"] == 1, "TotalBsmtSF"] = np.log(train["TotalBsmtSF"])
