## Structure of this code base
```
├── checkpoints
│   ├── best_detector.pth //best checkpoint
│   ├── detector.pth //last checkpoint
│   ├── training_log.txt //training log
├── data
│   ├── VOCdevkit_2007 //created after downloading data)
│   ├── voc2007.txt //annotation of train/val set
│   ├── voc2007test.txt // annotation of test set 
├── src
│   ├── config.py
│   ├── dataset.py
│   ├── eval_voc.py
│   ├── predict.py
│   ├── resnet_yolo.py
│   ├── yolo_loss.py
├── vis_results
│   ├── (visualized success / failure cases)
├── main.py //training code
├── vis.py //visualize detection
├── download_data.sh //script to download dataset
├── environment.yaml //environment config
└── README.md
```

## Preparation
### Set up environment
```
conda env create -f environment.yaml
conda activate yolo_env
```
### Download PASCAL VOC 2007 dataset
```
sh download_data.sh
```

## Demo
Run the following command:
```
python vis.py --checkpoint XX/XX/XX.pth --num_img XX --res_dir XX/XX
```
- `--checkpoint`: path to the checkpoint to load to the model, `default=./checkpoints/best_detector.pth`.
- `--num_img`: number of sample to generate, `default=1`.
- `--res_dir`: path to the directory to save results, `default=./vis_results`.

## Training
Run the following command:
```
python main.py --log_file XX/XX --checkpoint XX/XX/XX.pth --start_epoch XX --end_epoch XX
```
- `--log_file`: path to log file, `default=./checkpoints/training_log.txt`
- `--checkpoint`: path to pretrained model checkpoint, `default=None`. If not specified, the model will load `ResNet50_Weights.DEFAULT`.
- `--start_epoch`: Epoch to start training, `default=0`.
- `--end_epoch`: Epoch to stop training, `default=80`.