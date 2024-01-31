from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utility import get_dataset_path, get_dataframe_split
from dataset_features import features
import matplotlib.pyplot as plt

class RandomForestExtractor:
    def __init__(self, n_estimators=300, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.feature_importance = None

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        self.feature_importance = self.model.feature_importances_

    #if plt true return value to create chart, else only the top 15 most important features selected by random_forest
    def get_important_features(self, x_train, plt=False):
        features_and_importance = list(zip(x_train.columns, self.feature_importance))
        features_and_importance_sorted = sorted(features_and_importance, key=lambda x: x[1], reverse=True)
        features, importances_value = zip(*features_and_importance_sorted)
        if(plt):
            return features, importances_value
        else:
            print("hi")
            return list(features[:15])
    
    
pathDSTrain = get_dataset_path("TrainOnlyDoSProbe")
x_train , y_train = get_dataframe_split(pathDSTrain)

rf_features_extractor = RandomForestExtractor()
rf_features_extractor.fit(x_train, y_train)
important_features = rf_features_extractor.get_important_features(x_train)
print(important_features)

'''# create chart
plt.figure(figsize=(10, 6))
plt.bar(features, importances)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.xticks(rotation=45, ha='right')  # Rotare i nomi delle feature per una migliore leggibilità
plt.tight_layout()

# save chart as png
plt.savefig('random_forest_features_importance.png')'''


