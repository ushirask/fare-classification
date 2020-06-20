import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score


trainData = pd.read_csv('train.csv',index_col='tripid')
testData = pd.read_csv('test.csv',index_col='tripid')

trainData['label'] = [0 if x == "incorrect" else 1 for x in trainData['label']]

def calculateDistance(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance

trainData['distance'] = calculateDistance(trainData['drop_lat'],trainData['drop_lon'],trainData['pick_lat'],trainData['pick_lon'])
testData['distance'] = calculateDistance(testData['drop_lat'],testData['drop_lon'],testData['pick_lat'],testData['pick_lon'])

trainData['pickup_time'] = pd.to_datetime(trainData['pickup_time'], errors='coerce')
trainData['drop_time'] = pd.to_datetime(trainData['drop_time'], errors='coerce')
trainData['pickup_hour'] = trainData['pickup_time'].dt.hour
trainData['drop_hour'] = trainData['drop_time'].dt.hour
trainData['day'] = trainData['pickup_time'].dt.day


testData['pickup_time'] = pd.to_datetime(testData['pickup_time'], errors='coerce')
testData['drop_time'] = pd.to_datetime(testData['drop_time'], errors='coerce')
testData['pickup_hour'] = testData['pickup_time'].dt.hour
testData['drop_hour'] = testData['drop_time'].dt.hour
testData['day'] = testData['pickup_time'].dt.day

trainData = trainData.drop(['drop_lat','drop_lon','pick_lat','pick_lon','pickup_time','drop_time'], axis = 1)
testData = testData.drop(['drop_lat','drop_lon','pick_lat','pick_lon','pickup_time','drop_time'], axis = 1)

trainData.fillna(-999, inplace=True)
testData.fillna(-999,inplace=True)

y = trainData.label
X = trainData.drop('label',1)


train_features, test_features, train_targets, test_targets = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=42)

cat_fea=[7,8,9]
classifier=CatBoostClassifier(verbose=0, learning_rate=0.4,iterations=100, cat_features=cat_fea)
#classifier=CatBoostClassifier(verbose=0, learning_rate=0.4,iterations=100)
classifier.fit(train_features, train_targets) # fit the model for training data

# predict the 'target' for 'training data'
prediction_training_targets = classifier.predict(train_features)
self_accuracy = accuracy_score(train_targets, prediction_training_targets)
print("Accuracy for training data (self accuracy):", self_accuracy)
print("F1 self:",f1_score(train_targets,prediction_training_targets ))

# predict the 'target' for 'test data'
prediction_test_targets = classifier.predict(test_features)
test_accuracy = accuracy_score(test_targets, prediction_test_targets)
print("Accuracy for test data:", test_accuracy)
print("F1:",f1_score(test_targets,prediction_test_targets ))

classifier.fit(X,y)
predictions= classifier.predict(testData)

res = pd.DataFrame({'tripid': testData.index, 'prediction': predictions})
res.to_csv("results.csv", index=False)
