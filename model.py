import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv(r'C:\Users\raksh\OneDrive\Desktop\train.csv')
test = pd.read_csv(r'C:\Users\raksh\OneDrive\Desktop\test.csv')

# Fill missing values
train['Temperature'] = train['Temperature'].fillna(train['Temperature'].median())
train['Weather'] = train['Weather'].fillna('Sunny')
train['RoadType'] = train['RoadType'].fillna('Unknown')
test['Temperature'] = test['Temperature'].fillna(test['Temperature'].median())
test['Weather'] = test['Weather'].fillna('Sunny')
test['RoadType'] = test['RoadType'].fillna('Unknown')

# Extract hour and minute from timestamp
train['hour'] = train['timestamp'].apply(lambda x: int(x.split(':')[0]))
train['minute'] = train['timestamp'].apply(lambda x: int(x.split(':')[1]))
test['hour'] = test['timestamp'].apply(lambda x: int(x.split(':')[0]))
test['minute'] = test['timestamp'].apply(lambda x: int(x.split(':')[1]))

# Encode text columns
le = LabelEncoder()
for col in ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']:
    combined = pd.concat([train[col], test[col]])
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

features = ['geohash', 'day', 'hour', 'minute', 'RoadType', 
            'NumberofLanes', 'LargeVehicles', 'Landmarks', 
            'Temperature', 'Weather']

X_train = train[features]
y_train = train['demand']
X_test = test[features]

model = XGBRegressor(n_estimators=500, learning_rate=0.05, 
                     max_depth=8, random_state=42, n_jobs=-1)
model.fit(X_train, y_train, verbose=100)

predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, 1)

submission = pd.DataFrame({'Index': test['Index'], 'demand': predictions})
submission.to_csv(r'C:\Users\raksh\OneDrive\Desktop\submission.csv', index=False)
print("Done! Submission saved to Desktop.")