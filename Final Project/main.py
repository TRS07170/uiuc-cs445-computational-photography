import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.resnet_yolo import resnet50
from src.yolo_loss import YoloLoss
from src.dataset import VocDetectorDataset
from src.eval_voc import evaluate
from src.config import *

import collections
import argparse


def train(args, net, optimizer, criterion, train_loader, test_loader, device):
    best_test_loss = np.inf

    with open(args.log_file, 'a') as log_file:
        for epoch in tqdm(range(args.start_epoch, args.end_epoch)):
            net.train()
            
            # Update learning rate late in training
            cur_lr = learning_rate
            if epoch >= 50:
                cur_lr /= 10.0

            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr
            
            log_file.write('\n\nStarting epoch %d / %d\n' % (epoch + 1, args.end_epoch))
            log_file.write('Learning Rate for this epoch: {}\n'.format(cur_lr))
            
            total_loss = collections.defaultdict(int)
            
            for i, data in enumerate(train_loader):
                data = (item.to(device) for item in data)
                images, target_boxes, target_cls, has_object_map = data
                pred = net(images)
                loss_dict = criterion(pred, target_boxes, target_cls, has_object_map)
                for key in loss_dict:
                    total_loss[key] += loss_dict[key].item()
                
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                optimizer.step()
                
                if (i+1) % 50 == 0:
                    outstring = 'Epoch [%d/%d], Iter [%d/%d], Loss: ' % ((epoch+1, args.end_epoch, i+1, len(train_loader)))
                    outstring += ', '.join( "%s=%.3f" % (key[:-5], val / (i+1)) for key, val in total_loss.items() )
                    log_file.write(outstring + '\n')
            
            # evaluate the network on the test data
            if (epoch + 1) % 5 == 0:
                test_aps = evaluate(net, test_dataset_file=annotation_file_test, img_root=file_root_test, log_file=log_file)
                log_file.write(f"{epoch}, {test_aps}\n")
            with torch.no_grad():
                test_loss = 0.0
                net.eval()
                for i, data in enumerate(test_loader):
                    data = (item.to(device) for item in data)
                    images, target_boxes, target_cls, has_object_map = data
                    
                    pred = net(images)
                    loss_dict = criterion(pred, target_boxes, target_cls, has_object_map)
                    test_loss += loss_dict['total_loss'].item()
                test_loss /= len(test_loader)
            
            if best_test_loss > test_loss:
                best_test_loss = test_loss
                log_file.write('Updating best test loss: %.5f\n' % best_test_loss)
                torch.save(net.state_dict(),'./checkpoints/best_detector.pth')
            
            if (epoch+1) in [5, 10, 20, 30, 40]:
                torch.save(net.state_dict(),'./checkpoints/detector_epoch_%d.pth' % (epoch+1))

            torch.save(net.state_dict(),'./checkpoints/detector.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str, default="./checkpoints/training_log.txt", help='Path to pretrained model checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained model checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training')
    parser.add_argument('--end_epoch', type=int, default=80, help='Epoch to stop training')
    args = parser.parse_args()

    with open(args.log_file, 'a') as log_file:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_file.write(f"Device: {device}\n")
        if args.checkpoint is not None:
            log_file.write('Loading saved network from {}\n'.format(args.checkpoint))
            net = resnet50().to(device)
            net.load_state_dict(torch.load(args.checkpoint))
        else:
            log_file.write('Load pre-trained model\n')
            net = resnet50(pretrained=True).to(device)

        # Load training dataset
        train_dataset = VocDetectorDataset(root_img_dir=file_root_train,dataset_file=annotation_file_train,train=True, S=S)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
        log_file.write('Loaded %d train images\n' % len(train_dataset))

        # Load test dataset
        test_dataset = VocDetectorDataset(root_img_dir=file_root_test,dataset_file=annotation_file_test,train=False, S=S)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
        log_file.write('Loaded %d test images\n' % len(test_dataset))

        # Set up training tools
        criterion = YoloLoss(S, B, lambda_coord, lambda_noobj)
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    train(args, net, optimizer, criterion, train_loader, test_loader, device)


if __name__ == '__main__':
    main()