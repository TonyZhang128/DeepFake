import torch
import numpy as np
import os
import collections
import sys
sys.path.append('/home/zyn/DeepFake/dfdc_deepfake_challenge')

import pandas as pd
from tqdm import tqdm
from training.datasets.classifier_dataset import DeepFakeClassifierDataset
from torch.utils.data import DataLoader
from training.zoo import classifiers
from training.tools.config import load_config
from train_classifier import create_val_transforms

ckpt = "/home/zyn/DeepFake/dfdc_deepfake_challenge/weights/best_0.193.pth"
checkpoint = torch.load(ckpt, map_location='cuda:0')

conf = load_config('configs/b5.json')

model = classifiers.__dict__[conf['network']](encoder=conf['encoder']).to('cuda:0')

model.load_state_dict(checkpoint, strict=False)

batch_size = 20
data_val = DeepFakeClassifierDataset(mode="test",
                                    # fold=args.fold,
                                    padding_part=3,
                                    crops_dir='crops_test',
                                    data_path='/data/zyn/DeepFake',
                                    # folds_csv=args.folds_csv,
                                    label = 'pesudo_label.npy',
                                    transforms=create_val_transforms(conf["size"]),
                                    normalize=conf.get("normalize", None))
val_data_loader = DataLoader(data_val, batch_size=batch_size * 2, num_workers=16, shuffle=False,
                                pin_memory=False)

with torch.no_grad():
    data_val.reset(0, 438)
    test_dict = collections.OrderedDict()
    for sample in tqdm(val_data_loader):
        id = sample["id"].cuda()
        imgs = sample["image"].cuda()
        # img_names = sample["img_name"]
        # labels = sample["labels"].cuda().float()
        out = model(imgs)
        # labels = labels.cpu().numpy()
        # preds = torch.sigmoid(out).cpu().numpy()
        label = out > 0.5
        # for i in range(out.shape[0]):
        #     video, img_id = img_names[i].split("/")
        #     probs[video].append(preds[i].tolist())
        #     targets[video].append(labels[i].tolist())
        for i in range(len(id)):
            test_dict[id[i].item()] = label[i].item() if id[i].item() not in test_dict.keys() else print('error')
    order = sorted(test_dict.keys())
    result = [[id, test_dict[id]] for id in order]
    cow = ['test_'+str(item[0])+'.jpg' + '\t' + str(item[1]) for item in result]
    datas = {"Fnames label" : cow}
    
    # save to a csv
    Df = pd.DataFrame(datas, columns=["Fnames label"])
    Df.to_csv('/data/zyn/DeepFake/test.labels_' + ckpt.split('/')[-1].split('.pth')[0] + '.csv',header=True, index=False)


