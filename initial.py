import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

fedfiles = ["MORTGAGE30US.csv", "RRVRUSQ156N.csv", "CPIAUCSL.csv"]

#formats dates
dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fedfiles]

#Combines the datasets together into one dataframe
fedData = pd.concat(dfs, axis=1)

#Fills the values with the previous one to "forward fill" Ex: Nan in next box you copy the last non Nan value until you get a new non Nan value
fedData = fedData.ffill().dropna()

print(fedData)

zillowFiles = ["Metro_median_sale_price_uc_sfrcondo_week.csv", "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"]
dfs = [pd.read_csv(f) for f in zillowFiles]
#Picks the first row of data and cuts off the first 5 cols
dfs = [pd.DataFrame(df.iloc[0,5:]) for df in dfs]

#Difference in col dates as one is weekly data whereas one is monthly
#So I am stripping the date away to only leave the month
for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")

priceData = dfs[0].merge(dfs[1], on="month")
priceData.index = dfs[0].index

del priceData["month"]
priceData.columns = ["price", "value"]



#Since there is a difference in dates due to the fed releasing on tuesdays and Zillow releasing on sundays we need to account for it
fedData.index = fedData.index + timedelta(days=2)


#Merges both dataframes together and only has rows where the dates match
combinedData = fedData.merge(priceData, left_index=True, right_index=True)

combinedData.columns = ["interest", "vacancy", "cpi", "price", "value"]


print(combinedData.columns)

#Calculating price removing inflation (cpi) (sales price)
combinedData["adj_price"] = combinedData["price"] / combinedData["cpi"] * 100

#Calculating Value removing inflation(cpi) (Zillow calculated price)
combinedData["adj_value"] = combinedData["value"] / combinedData["cpi"] * 100

#Adds a column for adjusted price in 13 weeks
combinedData["next_quarter"] = combinedData["adj_price"].shift(-13)
combinedData.dropna(inplace=True)

#Returns boolean regarding whether the price increased (1 up 0 down)
combinedData["change"] = (combinedData["next_quarter"] > combinedData["adj_price"]).astype(int)

features = ["interest", "vacancy", "adj_price", "adj_value"]
target = "change"

START = 260
STEP = 52

def predict(train, test, features, target):
    rf = RandomForestClassifier(min_samples_split=10, random_state=1)
    rf.fit(train[features], train[target])
    preds = rf.predict(test[features])
    return preds

#Backtesting is checking to make sure that you are using only old data to predict the future at not new data to predict old ex using 2018 data to predict 2010
def backtest(data, features, target):
    all_preds = []
    for i in range(START, data.shape[0], STEP):
        train = combinedData.iloc[:i]
        test = combinedData.iloc[i:(i+STEP)]
        all_preds.append(predict(train, test, features, target))
    
    preds = np.concatenate(all_preds)
    return preds, accuracy_score(data.iloc[START:][target], preds)

preds, accuracy = backtest(combinedData, features, target)

print("Initial Model Accuracy: ", accuracy)

yearly = combinedData.rolling(52, min_periods=1).mean()

#Provides model with ideas for pricing trends by using ratios
yearly_ratios = [p + "_year" for p in features]
combinedData[yearly_ratios] = combinedData[features] / yearly[features]

preds, accuracy = backtest(combinedData, features + yearly_ratios, target)
print("Model Accuracy with price trend ratios: ", accuracy)


pred_match = (preds == combinedData[target].iloc[START:])

pred_match = pred_match.astype(object)

# Set the values to "green" and "red"
pred_match[pred_match == True] = "green"
pred_match[pred_match == False] = "red"

# Prepare the data for plotting
plot_data = combinedData.iloc[START:].copy()

# Plot the data
plot_data.reset_index().plot.scatter(x="index", y="adj_price", color=pred_match)
plt.show()

#Based on the plot the model suffers from sharp downturns during growth


rf = RandomForestClassifier(min_samples_split=10, random_state=1)
rf.fit(combinedData[features], combinedData[target])

result = permutation_importance(rf, combinedData[features], combinedData[target], n_repeats=10, random_state=1)
print(result["importances_mean"])
print(features)

#Most important featurse are adjusted value, adjusted price, interest rates, and vacancy rates