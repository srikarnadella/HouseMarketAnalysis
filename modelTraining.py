import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
import pickle

# Load and preprocess data
fedfiles = ["data/MORTGAGE30US.csv", "data/RRVRUSQ156N.csv", "data/CPIAUCSL.csv"]
dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fedfiles]
fedData = pd.concat(dfs, axis=1).ffill().dropna()

zillowFiles = ["data/Metro_median_sale_price_uc_sfrcondo_week.csv", "data/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"]
dfs = [pd.read_csv(f) for f in zillowFiles]
dfs = [pd.DataFrame(df.iloc[0, 5:]) for df in dfs]

for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")

priceData = dfs[0].merge(dfs[1], on="month")
priceData.index = dfs[0].index
del priceData["month"]
priceData.columns = ["price", "value"]

fedData.index = fedData.index + timedelta(days=2)
combinedData = fedData.merge(priceData, left_index=True, right_index=True)
combinedData.columns = ["interest", "vacancy", "cpi", "price", "value"]

combinedData["adj_price"] = combinedData["price"] / combinedData["cpi"] * 100
combinedData["adj_value"] = combinedData["value"] / combinedData["cpi"] * 100
combinedData["next_quarter"] = combinedData["adj_price"].shift(-13)
combinedData.dropna(inplace=True)
combinedData["change"] = (combinedData["next_quarter"] > combinedData["adj_price"]).astype(int)

features = ["interest", "vacancy", "adj_price", "adj_value"]
target = "change"
START = 260
STEP = 52

def predict(train, test, features, target, model):
    model.fit(train[features], train[target])
    preds = model.predict(test[features])
    return preds

def backtest(data, features, target, model):
    all_preds = []
    for i in range(START, data.shape[0], STEP):
        train = combinedData.iloc[:i]
        test = combinedData.iloc[i:(i + STEP)]
        all_preds.append(predict(train, test, features, target, model))
    
    preds = np.concatenate(all_preds)
    return preds, accuracy_score(data.iloc[START:][target], preds)

# Initial Random Forest model
rf = RandomForestClassifier(min_samples_split=10, random_state=1)
preds, accuracy = backtest(combinedData, features, target, rf)
print("Initial Model Accuracy (Random Forest):", accuracy)

# Adding yearly ratios
yearly = combinedData.rolling(52, min_periods=1).mean()
yearly_ratios = [p + "_year" for p in features]
combinedData[yearly_ratios] = combinedData[features] / yearly[features]

# Backtesting with yearly ratios
preds, accuracy = backtest(combinedData, features + yearly_ratios, target, rf)
print("Model Accuracy with price trend ratios (Random Forest):", accuracy)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
preds, accuracy = backtest(combinedData, features + yearly_ratios, target, lr)
print("Model Accuracy with price trend ratios (Logistic Regression):", accuracy)

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=1)
preds, accuracy = backtest(combinedData, features + yearly_ratios, target, gb)
print("Model Accuracy with price trend ratios (Gradient Boosting):", accuracy)

# Feature importance with RandomForest
rf.fit(combinedData[features], combinedData[target])
result = permutation_importance(rf, combinedData[features], combinedData[target], n_repeats=10, random_state=1)
feature_importances = dict(zip(features, result["importances_mean"]))
print("Feature importances (Random Forest):", feature_importances)

# Save feature importances
with open('feature_importances.pkl', 'wb') as f:
    pickle.dump(feature_importances, f)

# Grid Search for RandomForest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=1), param_grid, cv=5, scoring='accuracy')
grid_search.fit(combinedData[features + yearly_ratios], combinedData[target])
print("Best parameters (Random Forest):", grid_search.best_params_)

best_rf = grid_search.best_estimator_
preds, accuracy = backtest(combinedData, features + yearly_ratios, target, best_rf)
print("Best Model Accuracy with price trend ratios (Random Forest):", accuracy)

# Save results for analysis
combinedData.to_csv('combined_data.csv')
np.save('preds.npy', preds)
