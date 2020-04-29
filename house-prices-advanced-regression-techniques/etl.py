import dabl
import numpy as np
import pandas as pd


def analyze_types(train):
    # analyze feature types
    types = dabl.detect_types(train)

    continuous_features = types[types["continuous"]].index.values.tolist()

    categorical_features = types[types["categorical"]].index.values.tolist()
    categorical_features += types[types["low_card_int"]].index.values.tolist()

    useless_features = types[types["useless"]].index.values.tolist()
    return continuous_features, categorical_features, useless_features


def prepare_data():
    # load and shuffle train and test
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    test_id = test["Id"]
    train = train.drop(["Id"], axis=1)
    test = test.drop(["Id"], axis=1)
    train = train.sample(frac=1.0, random_state=69)
    test = test.sample(frac=1.0, random_state=69)

    # analyze drop useless features
    continuous_features, categorical_features, useless_features = analyze_types(train)
    train = train.drop(useless_features, axis=1)
    test = test.drop(useless_features, axis=1)

    # handle missing data
    total = train.isnull().sum().sort_values(ascending=False)
    percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=["total", "percent"])
    missing_data["percent"] *= 100
    missing_data = missing_data[missing_data["total"] > 0]

    # drop missing columns since they don't co relate much with target
    train = train.drop((missing_data[missing_data["total"] > 1]).index, 1)
    train = train.drop(train.loc[train["Electrical"].isnull()].index)

    test = test.drop((missing_data[missing_data["total"] > 1]).index, 1)
    test = test.drop(test.loc[test["Electrical"].isnull()].index)

    # log transform for skewed columns
    train = train.drop(train.loc[train["Electrical"].isnull()].index)
    test = test.drop(test.loc[test["Electrical"].isnull()].index)

    # analyzing dist plots and correcting skewness by log
    train["SalePrice"] = np.log1p(train["SalePrice"])

    # do combined transformations for train/test
    df = pd.concat([train, test])
    df = df.drop(["SalePrice"], axis=1)
    df = df.reset_index(drop=True)

    # log transformations
    df["GrLivArea"] = np.log1p(df["GrLivArea"])
    df["GrLivArea"] = np.log1p(df["GrLivArea"])
    # handling log 0 for columns with 0 vals
    df["TotalBsmtSF"] = np.log1p(df["TotalBsmtSF"])

    # re analyze
    continuous_features, categorical_features, useless_features = analyze_types(train)

    # fix data type of categorical columns
    for col in categorical_features:
        df[col] = df[col].astype(str)

    # scale continuous features
    stats = train[continuous_features].describe()
    for var in continuous_features:
        if var == "SalePrice":
            continue
        # df[var] = 1.0 * df[var] - stats[var]["mean"]
        # df[var] = df[var] / stats[var]["std"]
        df[var] = np.log1p(df[var])

    # one hot encoding for dummy variables
    df = pd.get_dummies(df, prefix=categorical_features)

    for col in df.columns:
        df[col] = df[col].astype("float")

    # separate train/test
    m = len(train)

    print("Shapes")
    print("Train", df.iloc[:m].values.shape)
    print("Test", df.iloc[m:].values.shape)
    print("y", train["SalePrice"].shape)

    # return (train[:m], train["SalePrice"]), (train[m:], test_id)
    return (df.iloc[:m].values, train[["SalePrice"]].values), (df.iloc[m:].values, test_id)


def prepare_submission(test_id, y_hat):
    # empty df
    df = pd.DataFrame()

    # inverse log transform
    y_hat = np.expm1(y_hat)

    df["Id"] = test_id
    df["SalePrice"] = y_hat
    df.to_csv("submission.csv", index=False)
