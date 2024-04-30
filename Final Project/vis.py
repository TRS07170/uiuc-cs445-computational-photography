import cv2
import random
import os
import torch
import argparse
from tqdm import tqdm

from src.resnet_yolo import resnet50
from src.predict import predict_image
from src.config import *
from src.dataset import VocDetectorDataset


def vis():
    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_detector.pth', help='Path to detector checkpoint')
    parser.add_argument('--num_img', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--res_dir', type=str, default='./vis_results', help='Path to directory to save results')
    args = parser.parse_args()

    # load test dataset
    test_dataset = VocDetectorDataset(root_img_dir=file_root_test,dataset_file=annotation_file_test,train=False, S=S)
    # print('Loaded %d test images\n' % len(test_dataset))

    # load detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet50().to(device)
    net.load_state_dict(torch.load(args.checkpoint))
    net.eval()

    # select random image from test set
    image_list = random.choices(test_dataset.fnames, k=args.num_img)
    print('Predicting...')
    for image_name in tqdm(image_list):
        image = cv2.imread(os.path.join(file_root_test, image_name))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result = predict_image(net, image_name, root_img_directory=file_root_test)
        for left_up, right_bottom, class_name, _, prob in result:
            color = COLORS[VOC_CLASSES.index(class_name)]
            cv2.rectangle(image, left_up, right_bottom, color, 2)
            label = class_name + str(round(prob, 2))
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                          color, -1)
            cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

        # plt.figure(figsize = (15,15))
        # plt.imshow(image)
        cv2.imwrite(os.path.join(args.res_dir, image_name), image)


if __name__ == '__main__':
    vis()