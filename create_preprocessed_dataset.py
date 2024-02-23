import os
import numpy as np
import pandas as pd
from dataset_features import features, features_to_encode, rf_important_features, shap_snn_important_features, shap_rnn_important_features, shap_drnn_important_features
from dataset_features import shap_snn_impactful_features, shap_rnn_impactful_features, shap_drnn_impactful_features
from attacks_categories import u2r_attacks, r2l_attacks, dos_attacks, probe_attacks
from sklearn.preprocessing import MinMaxScaler
from utility import get_dataset_path

#this func create a dataset encoded, normalized and without u2r-r2l attack (to reduce overfitting) with all 41 features
def create_dataset_only_dos_probe_attack(datasetPath, pathWriteNewDS, datasetType):

    #read data
    df = pd.read_csv(datasetPath)

    #set columns name
    df.columns = features

    #deleted u2r and r2l attacks rows
    df = df[~df['labels'].isin(u2r_attacks) & ~df['labels'].isin(r2l_attacks)]

    #set labels in dos_attac or prob_attack based on attack categories
    df.loc[df['labels'].isin(dos_attacks), 'labels'] = "dos_attack"
    df.loc[df['labels'].isin(probe_attacks), 'labels'] = "probe_attack"

    #delete column level
    df = df.drop("level", axis=1)

    #encode 'protocol_type','service' and 'flag'
    df[features_to_encode] = df[features_to_encode].astype('category').apply(lambda x: x.cat.codes)

    #encode labels
    mapping_for_labels = {'normal': 0, 'dos_attack': 1, 'probe_attack': 2}
    df['labels'] = pd.Categorical(df['labels'], categories=mapping_for_labels.keys()).codes

    #handling outlier of features src_bytes and dst_bytes with loge
    df['src_bytes'] = df['src_bytes'].apply(lambda x: np.log1p(x) if x > 0 else 0)
    df['dst_bytes'] = df['dst_bytes'].apply(lambda x: np.log1p(x) if x > 0 else 0)

    #setting the features to scale (all except labels)
    columns_to_scale = df.columns[:-1]
    columns_to_exclude = ['labels']
    features_to_scale = df[columns_to_scale]
    features_to_exclude = df[columns_to_exclude]
    
    #normalize data between 0 and 1
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(features_to_scale), columns=columns_to_scale)

    #reset indices before concatenating to avoid NaN values
    df.reset_index(drop=True, inplace=True)
    features_to_exclude.reset_index(drop=True, inplace=True)

    df = pd.concat([df, features_to_exclude], axis=1)
    df = df[df['labels'] != -1]

    #conversion of dataframe to txt
    df.to_csv(pathWriteNewDS + "KDD" + datasetType + "OnlyDoSProbe", index=False)
    print("dataset creato con successo")

#this func create a dataset with only most important features selected by Random Forest or by shap
#dataset to pass: OnlyDoSProbe
def create_dataset_with_important_features(datasetPath, pathWriteNewDS, dataset_type, extractor_name, important_features):
    df = pd.read_csv(datasetPath)
    df.columns = features[ : -1]
    df = df[important_features]
    df.to_csv(pathWriteNewDS + "KDD" + dataset_type + extractor_name + "ImpactfulFeatures", index=False)
    print("dataset creato con successo")

#this func create a dataset with 0-1 lables for binary classification, dataset to pass: OnlyDoSProbe or ImportantFeatures
def create_dataset_with_binary_labels(datasetPath, pathWriteNewDS, dataset_type, category):
    df = pd.read_csv(datasetPath)
    last_column = df.columns[-1]
    df[last_column] = df[last_column].replace(2, 1)
    df.to_csv(pathWriteNewDS + "KDD" + dataset_type + "Binary" + category , index=False)


'''pathRawTrain = get_dataset_path("Train")
pathRawTest = get_dataset_path("Test")'''

pathODPTrain = get_dataset_path("TrainOnlyDoSProbe")
pathODPTest = get_dataset_path("TestOnlyDoSProbe")

pathIFTrain = get_dataset_path("TrainImportantFeatures")
pathIFTest = get_dataset_path("TestImportantFeatures")

#get current work directory
cwd = os.getcwd()

#set dataset path where to write modified dataset
pathWriteNewDS = cwd + "\\NSL-KDD\\"

'''create_dataset_only_dos_probe_attack(pathRawTrain,pathWriteNewDS, "Train")
create_dataset_only_dos_probe_attack(pathRawTest,pathWriteNewDS, "Test")'''




'''create_dataset_with_important_features(pathODPTrain,pathWriteNewDS, "Train", "RF", rf_important_features)
create_dataset_with_important_features(pathODPTest,pathWriteNewDS, "Test", "RF", rf_important_features)'''

'''create_dataset_with_important_features(pathODPTrain,pathWriteNewDS, "Train", "SHAPSNN", shap_snn_important_features)
create_dataset_with_important_features(pathODPTest,pathWriteNewDS, "Test", "SHAPSNN", shap_snn_important_features)

create_dataset_with_important_features(pathODPTrain,pathWriteNewDS, "Train", "SHAPRNN", shap_rnn_important_features)
create_dataset_with_important_features(pathODPTest,pathWriteNewDS, "Test", "SHAPRNN", shap_rnn_important_features)

create_dataset_with_important_features(pathODPTrain,pathWriteNewDS, "Train", "SHAPDRNN", shap_drnn_important_features)
create_dataset_with_important_features(pathODPTest,pathWriteNewDS, "Test", "SHAPDRNN", shap_drnn_important_features)'''

'''create_dataset_with_important_features(pathODPTrain,pathWriteNewDS, "Train", "SHAPSNN", shap_snn_impactful_features)
create_dataset_with_important_features(pathODPTest,pathWriteNewDS, "Test", "SHAPSNN", shap_snn_impactful_features)

create_dataset_with_important_features(pathODPTrain,pathWriteNewDS, "Train", "SHAPRNN", shap_rnn_impactful_features)
create_dataset_with_important_features(pathODPTest,pathWriteNewDS, "Test", "SHAPRNN", shap_rnn_impactful_features)

create_dataset_with_important_features(pathODPTrain,pathWriteNewDS, "Train", "SHAPDRNN", shap_drnn_impactful_features)
create_dataset_with_important_features(pathODPTest,pathWriteNewDS, "Test", "SHAPDRNN", shap_drnn_impactful_features)'''




'''create_dataset_with_binary_labels(pathODPTrain,pathWriteNewDS, "Train", "OnlyDoSProbe")
create_dataset_with_binary_labels(pathODPTest,pathWriteNewDS, "Test", "OnlyDoSProbe")

create_dataset_with_binary_labels(pathIFTrain,pathWriteNewDS, "Train", "ImportantFeatures")
create_dataset_with_binary_labels(pathIFTest,pathWriteNewDS, "Test", "ImportantFeatures")'''

