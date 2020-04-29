import numpy as np
from etl import prepare_data, prepare_submission
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# load and split train/dev/test
(X_train, y_train), (X_test, test_id) = prepare_data()
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=69)

# run without hyperparams for fscore calculations
model = XGBRegressor()
model.fit(X_train, y_train)

y_hat = model.predict(X_dev)
mae = mean_absolute_error(np.expm1(y_dev), np.expm1(y_hat))
print("Mae: {}".format(mae))

thresholds = np.sort(model.feature_importances_)
thresholds = np.unique(thresholds)
threshold = 0
best_mae = mae

for thresh in thresholds[:50]:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBRegressor()
    selection_model.fit(select_X_train, y_train)

    # eval model
    select_X_dev = selection.transform(X_dev)
    y_hat = selection_model.predict(select_X_dev)
    mae = mean_absolute_error(np.expm1(y_dev), np.expm1(y_hat))

    # select thresh with least mae
    if mae > best_mae:
        best_mae = mae
        threshold = thresh
    print("Thresh={}, Features={}, MAE: {}".format(thresh, select_X_train.shape[1], mae))

print()
print("Best Threshold: ", threshold)

selection = SelectFromModel(model, threshold=threshold, prefit=True)
X_train = selection.transform(X_train)
X_dev = selection.transform(X_dev)
X_test = SelectFromModel(model, threshold=threshold, prefit=True)

# final model
model = XGBRegressor(
    colsample_bytree=0.4,
    gamma=0,
    learning_rate=0.07,
    max_depth=3,
    min_child_weight=1.5,
    n_estimators=10000,
    reg_alpha=0.75,
    reg_lambda=0.45,
    subsample=0.6,
    seed=42
)

model.fit(X_train, y_train)

# validation accuracy
y_hat = model.predict(X_dev)
mae = mean_absolute_error(np.expm1(y_dev), np.expm1(y_hat))
print(mae)

# final predictions
y_hat = model.predict(X_test)
prepare_submission(test_id, y_hat)
