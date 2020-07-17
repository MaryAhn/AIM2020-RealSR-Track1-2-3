# AIM 2020 Real Image Super-Resolution Challenge (Track1, Track2, Track3)
- Keon-Hee Ahn, MCML group, Yonsei University, South Korea
- email: khahn196@gmail.com
- If you have any questions, please contact me through above email

## Requirements
- pytorch-gpu 
- numpy
- math
- os
- argparse
- copy
- queue
- threading
- importlib
- time
- cv2
- torch.nn
- torch.optim
- torch.nn.functional

## Folders
- code
	- dataloaders: folder containing dataloader
		- dataloader_multiscale.py: for loading data
	- utils: folder containing utlity
		- image_utils.py: some functions for processing data
	- model: folder which contain models
		- base.py: base for model
		- mdsr_mod5: our model
	- train_queued.py: for training models
	- get_sr.py: for acquring result images

- data: RealSR testing data for each track 
	- TestX2: test for track 1
	- TestX3: test for track 2
	- TestX4: test for track 3

- experiments: trained model checkpoint for each track 
	- Track1
	- Track2
	- Track3

## How To Use
1. --cuda_device=0: It is device selection in case of multiple GPU. If you have only one GPU, just set it to 0.
2. --input_path, --output_path, --restore_path: you have to change 'your_directory' part in each argument to suit your environment.
3. After modifying above three arguemtns, run the following commands for each Track. Then, the result images will be created. (ex, challenge\experiments\Track1\results\)

### Track 1:
python get_sr.py --model=mdsr_mod5 --restore_path=your_directory\challenge\experiments\Track_1\model_200000.pth --input_path=your_directory\challenge\data\TestLRX2\TestLR --scale=2 --edsr_res_blocks=80 --output_path=your_directory\challenge\experiments\Track_1\results --cuda_device=0 --chop_forward 

### Track 2:
python get_sr.py --model=mdsr_mod5 --restore_path=your_directory\challenge\experiments\Track_2\model_200000.pth --input_path=your_directory\challenge\data\TestLRX3\TestLR --scale=3 --edsr_res_blocks=80 --output_path=your_directory\challenge\experiments\Track_2\results --cuda_device=0  --self_ensemble --chop_forward 

### Track 3:
python get_sr.py --model=mdsr_mod5 --restore_path=your_directory\challenge\experiments\Track_3\model_200000.pth --input_path=your_directory\challenge\data\TestLRX4\TestLR --scale=4 --edsr_res_blocks=80 --output_path=your_directory\challenge\experiments\Track_3\results --cuda_device=0  --self_ensemble --chop_forward 
