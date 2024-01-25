import os
import numpy as np
import pandas as pd
from dataset_features import features, features_to_encode
from attacks_categories import u2r_attacks, r2l_attacks, dos_attacks, probe_attacks
from sklearn.preprocessing import MinMaxScaler
from utility import get_dataset_path

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

    #delete row level
    df = df.drop("level", axis=1)

    #encode 'protocol_type','service' and 'flag'
    df[features_to_encode] = df[features_to_encode].astype('category').apply(lambda x: x.cat.codes)

    #encode labels
    custom_mapping_for_labels = {'normal': 0, 'dos_attack': 1, 'probe_attack': 2}
    df['labels'] = pd.Categorical(df['labels'], categories=custom_mapping_for_labels.keys()).codes

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

    # Reset indices before concatenating to avoid NaN values
    df.reset_index(drop=True, inplace=True)
    features_to_exclude.reset_index(drop=True, inplace=True)

    df = pd.concat([df, features_to_exclude], axis=1)

    #conversion of dataframe to txt
    df.to_csv(pathWriteNewDS + "KDD" + datasetType + "OnlyDoSProbe", index=False)


#get current work directory
cwd = os.getcwd()

#set dataset train and test path
pathDSTrain = get_dataset_path("Train")
pathDSTest = get_dataset_path("Test")

#set dataset path where to write modified dataset
pathWriteNewDS = cwd + "\\NSL-KDD\\"

create_dataset_only_dos_probe_attack(pathDSTrain,pathWriteNewDS, "Train")
create_dataset_only_dos_probe_attack(pathDSTest,pathWriteNewDS, "Test")

