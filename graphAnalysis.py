import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
combinedData = pd.read_csv('combined_data.csv', index_col=0, parse_dates=True)
preds = np.load('preds.npy')

# Load feature importances
with open('feature_importances.pkl', 'rb') as f:
    feature_importances = pickle.load(f)

START = 260
STEP = 52

# Visualization of predictions
pred_match = (preds == combinedData['change'].iloc[START:])
pred_match = pred_match.astype(object)
pred_match[pred_match == True] = "green"
pred_match[pred_match == False] = "red"

plot_data = combinedData.iloc[START:].copy()
plot_data.reset_index().plot.scatter(x="index", y="adj_price", color=pred_match)
plt.title("Prediction Match Scatter Plot")
plt.xlabel("Date")
plt.ylabel("Adjusted Price")
plt.show()

# Time series plots
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
plt.plot(combinedData.index, combinedData['interest'], label='Interest Rate')
plt.title('Interest Rates Over Time')
plt.xlabel('Date')
plt.ylabel('Interest Rate')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(combinedData.index, combinedData['vacancy'], label='Vacancy Rate')
plt.title('Vacancy Rates Over Time')
plt.xlabel('Date')
plt.ylabel('Vacancy Rate')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(combinedData.index, combinedData['cpi'], label='CPI')
plt.title('CPI Over Time')
plt.xlabel('Date')
plt.ylabel('CPI')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(combinedData.index, combinedData['adj_price'], label='Adjusted Price')
plt.plot(combinedData.index, combinedData['adj_value'], label='Adjusted Value')
plt.title('Adjusted Prices and Values Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Price/Value')
plt.legend()
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = combinedData[['interest', 'vacancy', 'cpi', 'adj_price', 'adj_value']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# Distribution plots
plt.figure(figsize=(14, 8))
for i, col in enumerate(['interest', 'vacancy', 'cpi', 'adj_price', 'adj_value']):
    plt.subplot(2, 3, i+1)
    sns.histplot(combinedData[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Trend analysis
combinedData[['adj_price', 'adj_value']].plot(figsize=(14, 8))
plt.title('Trends in Adjusted Prices and Values')
plt.xlabel('Date')
plt.ylabel('Adjusted Price/Value')
plt.legend()
plt.show()

# Train model for future predictions
future_steps = 52  # Predict for 1 year into the future
rf = RandomForestClassifier(min_samples_split=10, random_state=1)
features = ['interest', 'vacancy', 'adj_price', 'adj_value']
yearly_ratios = [p + '_year' for p in features]
rf.fit(combinedData[features + yearly_ratios], combinedData['change'])

# Create future data
last_row = combinedData.iloc[-1]
future_dates = pd.date_range(start=combinedData.index[-1], periods=future_steps + 1, freq='W')
future_data = pd.DataFrame(index=future_dates[1:], columns=combinedData.columns)
future_data[['interest', 'vacancy', 'cpi']] = combinedData[['interest', 'vacancy', 'cpi']].iloc[-1].values
future_data['price'] = combinedData['price'].iloc[-1] * (1 + np.random.normal(0, 0.02, size=future_steps))
future_data['value'] = combinedData['value'].iloc[-1] * (1 + np.random.normal(0, 0.02, size=future_steps))
future_data['adj_price'] = future_data['price'] / future_data['cpi'] * 100
future_data['adj_value'] = future_data['value'] / future_data['cpi'] * 100

# Calculate future yearly ratios
yearly = combinedData.rolling(52, min_periods=1).mean().iloc[-1]
future_data[yearly_ratios] = future_data[features] / yearly[features]

# Predict future changes
future_preds = rf.predict(future_data[features + yearly_ratios])
future_data['predicted_change'] = future_preds
future_data['predicted_adj_price'] = future_data['adj_price'].shift(13) * (1 + future_data['predicted_change'] * 0.01)

# Plot future predictions
plt.figure(figsize=(14, 8))
plt.plot(combinedData.index, combinedData['adj_price'], label='Historical Adjusted Price')
plt.plot(future_data.index, future_data['predicted_adj_price'], label='Predicted Adjusted Price')
plt.title('Historical and Predicted Adjusted Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Price')
plt.legend()
plt.show()

# Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(list(feature_importances.keys()), list(feature_importances.values()))
plt.xlabel("Feature Importance")
plt.title("Feature Importance from Random Forest")
plt.show()
