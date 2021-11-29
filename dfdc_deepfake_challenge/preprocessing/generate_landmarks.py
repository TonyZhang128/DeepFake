import argparse
import os
from functools import partial
from multiprocessing.pool import Pool



os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from tqdm import tqdm

import sys
sys.path.append('/home/zyn/DeepFake/dfdc_deepfake_challenge')

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from preprocessing.utils import get_original_video_paths, get_original_img_paths

from PIL import Image
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np

detector = MTCNN(margin=0, thresholds=[0.65, 0.75, 0.75], device="cpu")


def save_landmarks(ori_id, root_dir):
    # ori_id = ori_id[:-4]
    # ori_dir = os.path.join(root_dir, "crops", ori_id)
    # landmark_dir = os.path.join(root_dir, "landmarks", ori_id)
    # os.makedirs(landmark_dir, exist_ok=True)
    ori_path = ori_id
    landmark_dir = os.path.join(root_dir, "landmarks")
    # for frame in range(320):
    #     if frame % 10 != 0:
    #         continue
    #     for actor in range(2):
    #         image_id = "{}_{}.png".format(frame, actor)
    #         landmarks_id = "{}_{}".format(frame, actor)
    #         ori_path = os.path.join(ori_dir, image_id)
    #         landmark_path = os.path.join(landmark_dir, landmarks_id)
    id = ori_path.split('/')[-1][:-4]
    landmark_path = os.path.join(landmark_dir, id+'.npy')
    if os.path.exists(ori_path):
        try:
            image_ori = cv2.imread(ori_path, cv2.IMREAD_COLOR)[...,::-1]
            img = Image.fromarray(image_ori)
            batch_boxes, conf, landmarks = detector.detect(img, landmarks=True)
            if landmarks is not None:
                landmarks = np.around(landmarks[0]).astype(np.int16)
                np.save(landmark_path, landmarks)
                # return landmarks
        except Exception as e:
            print(e)
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract image landmarks")
    parser.add_argument("--root-dir", help="root directory", default="/data/zyn/DeepFake")
    parser.add_argument("--landmarks_dir", default="/data/zyn/DeepFake/landmarks")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ids = get_original_img_paths(args.root_dir)
    os.makedirs(os.path.join(args.root_dir, "landmarks"), exist_ok=True)
    with Pool(processes=os.cpu_count()) as p:
        with tqdm(total=len(ids)) as pbar:
            func = partial(save_landmarks, root_dir=args.root_dir)
            for v in p.imap_unordered(func, ids):
                pbar.update()


if __name__ == '__main__':
    main()
