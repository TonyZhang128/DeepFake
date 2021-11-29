import csv
import os
from albumentations.augmentations.transforms import ToSepia
import numpy as np
import pandas as pd

def landmarks():
    landmarks = []
    landmarks_path = '/data/zyn/DeepFake/landmarks'
    for landmark_np in os.listdir(landmarks_path):
        name = landmark_np.split('_')[-1].split('.')[0]
        landmark = np.load(os.path.join(landmarks_path, landmark_np))
        row = np.array([int(name), landmark])
        landmarks.append(row)
    print(len(landmarks))
    np.save('/data/zyn/DeepFake/landmarks.npy',np.array(landmarks))

def label():
    labels = []
    label_path = "/data/zyn/DeepFake/train.labels.csv"
    # label_csv_path = os.path.join(root_dir,'train.labels.csv')
    csv = pd.read_csv(label_path, encoding='GBK')
    for i, row in csv.iterrows():
        label = row[0].split('\t')
        id = int(label[0].split('.')[0].split('_')[-1])
        if id != i: print('error\n')
        info = np.array([id,int(label[-1])])
        labels.append(info)
    np.save('/data/zyn/DeepFake/label.npy', np.array(labels))

def pesudo_test_label():
    labels = []
    test_path = "/data/zyn/DeepFake/image/test"
    imgs = os.listdir(test_path)
    for img in imgs:
        img = int(img.split('.')[0].split('_')[-1])
        labels.append(img)
    np.save('/data/zyn/DeepFake/pesudo_label.npy', np.array(labels)) 

pesudo_test_label()