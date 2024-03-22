import argparse
import os
import time
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from cifar import dataset
from cifar import model
from cifar import resnet
from cifar import DenseNet
from utee import hook

from robustbench.data import load_cifar10c
from torchvision import transforms

#from IPython import embed
from datetime import datetime
from subprocess import call
parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='cifar10|cifar100')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--decreasing_lr', default='200,250', help='decreasing strategy')
parser.add_argument('--wl_weight', default=2)
parser.add_argument('--wl_grad', default=8)
parser.add_argument('--wl_activate', default=8)
parser.add_argument('--wl_error', default=8)
parser.add_argument('--inference', default=1)
parser.add_argument('--onoffratio', default=10)
parser.add_argument('--cellBit', default=1)
parser.add_argument('--subArray', default=128)
parser.add_argument('--ADCprecision', default=5)
parser.add_argument('--vari', default=0)
parser.add_argument('--t', default=0)
parser.add_argument('--v', default=0)
parser.add_argument('--detect', default=0)
parser.add_argument('--target', default=0)
parser.add_argument('--model', default='VGG8', help='specifying DNN model')
parser.add_argument('--corrupted', type=int, default=0, help='whether apply Cifar10-C to test')

current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()
args.wl_weight = 8
args.wl_grad = 8
args.wl_error = 8
args.wl_activate = 8
args.batch_size = 1000
args.inference = 2            # set to run inference simulation
# Hardware Properties
args.subArray = 128           # size of subArray (e.g. 128*128)
args.ADCprecision = 6         # ADC precision (e.g. 5-bit)
args.cellBit = 8              # cell precision (e.g. 4-bit/cell)
args.onoffratio = 30          # device on/off ratio (e.g. Gmax/Gmin = 3)
# if do not run the device retention / conductance variation effects, set args.vari=0, args.v=0
args.vari = 0.                 # conductance variation (e.g. 0.1 standard deviation to generate random variation)
args.t = 0                    # retention time
args.v = 0                    # drift coefficient
args.detect = 1               # if 1, fixed-direction drift, if 0, random drift
args.target = 0.5             # drift target for fixed-direction drift
args.mode = 'WAGE'			  # specifying WAGE quantization 
# args.corrupted = int(args.corrupted)

# args.corruption = ["motion_blur", "snow", "pixelate",
#                           "defocus_blur", "brightness", "fog",
#                           "zoom_blur", "frost", 
#                           "jpeg_compression", "elastic_transform"]


args.corruption = ["motion_blur"]

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
	logger('{}: {}'.format(k, v))
logger("========================================")

# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)



if args.model == 'VGG8':
	# model_path = r'../model_files/VGG8_seed.pth'
	model_path = r'../model_files/vgg8_onoff30_c10.pth'
elif args.model == 'DN40':
	model_path = r'../model_files/DenseNet40_seed.pth'
elif args.model == 'RN20':
	model_path = r'../model_files/rn20_seed.pth'
else:
	raise ValueError(f'Model0-{args.model} not supported')

pretrained = model_path
# pretrained = None

if args.model == 'VGG8':
	modelCF = model.cifar10(args = args, logger=logger, pretrained = pretrained)
elif args.model == 'DN40':
	modelCF = DenseNet.densenet40(args = args, logger=logger, pretrained = pretrained)
elif args.model == 'RN20':
	modelCF = resnet.resnet20(args = args, logger=logger, pretrained = pretrained)
else:
	raise ValueError(f'Model0-{args.model} not supported')

# data loader and model
# assert args.type in ['cifar10', 'cifar100'], args.type
if args.corrupted == 1:
	trans = transforms.Compose([
		transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))
	])

	num_sample = 1000
	corruption_types = args.corruption
	total_num_img = num_sample * len(corruption_types)
	print(f'----------------- Applying cifar10c dataset, number of corruption: {len(corruption_types)}\n')
	# for corruption in corruption_types:
	# 	_correct_this_type = 0
	# 	_data, _target = load_cifar10c(n_examples=num_sample, corruptions = [corruption], severity = 5) 
	# 	_data = trans(_data)
	# 	_data = torch.split(_data, args.batch_size)
	# 	_target = torch.split(_target, args.batch_size)
else:
	train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1)


print(args.cuda)
if args.cuda:
	modelCF.cuda()
best_acc, old_file = 0, None
t_begin = time.time()
# ready to go
modelCF.eval()
test_loss = 0
correct = 0
acc = 0


if args.corrupted == 0:		# inference original Cifar10
	for i, (data, target) in enumerate(test_loader):
		if i==0:
			hook_handle_list = hook.hardware_evaluation(modelCF,args.wl_weight,args.wl_activate,0)
		indx_target = target.clone()
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		with torch.no_grad():
			data, target = Variable(data), Variable(target)
			output = modelCF(data)
			test_loss += F.cross_entropy(output, target).data
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.cpu().eq(indx_target).sum()
		if i==0:
			hook.remove_hook_list(hook_handle_list)

	test_loss = test_loss / len(test_loader)  # average over number of mini-batch
	acc = 100. * correct / len(test_loader.dataset)

	accuracy = acc.cpu().data.numpy()

elif args.corrupted == 1:		# testing against cifar10-c

	acc_per_type = []
	for corruption in corruption_types:
		_correct_this_type = 0
		_data, _target = load_cifar10c(n_examples=num_sample, corruptions = [corruption], severity = 5, data_dir = '../../DNN_NS2.1_legacy/Training_pytorch/data') 
		_data = trans(_data)
		_data = torch.split(_data, args.batch_size)
		_target = torch.split(_target, args.batch_size)

		for i in range(len(_data)):
			data = _data[i]
			target = _target[i]
			if i==0:
				hook_handle_list = hook.hardware_evaluation(modelCF,args.wl_weight,args.wl_activate,0)
			indx_target = target.clone()
			if args.cuda:
				data, target = data.cuda(), target.cuda()
			with torch.no_grad():
				data, target = Variable(data), Variable(target)
				output = modelCF(data)
				test_loss += F.cross_entropy(output, target).data
				pred = output.data.max(1)[1]  # get the index of the max log-probability
				tmp = pred.cpu().eq(indx_target).sum()
				# print(f'batch-wise correct prediction (batchsize= {args.batch_size}): {tmp}')
				correct += pred.cpu().eq(indx_target).sum()
				_correct_this_type += pred.cpu().eq(indx_target).sum()
			if i==0:
				hook.remove_hook_list(hook_handle_list)

		acc_per_type.append(float(_correct_this_type)/num_sample)
		print(f' --- corruption: {corruption}, accuracy: [{_correct_this_type}]/[{num_sample}]  ({acc_per_type[-1]}) --- ')

		test_loss /= total_num_img
		acc = 100. * correct / total_num_img
		accuracy = acc.cpu().data.numpy()

print(" --- Hardware Properties --- ")
print("subArray size: ")
print(args.subArray)
print("ADC precision: ")
print(args.ADCprecision)
print("cell precision: ")
print(args.cellBit)
print("on/off ratio: ")
print(args.onoffratio)
print("variation: ")
print(args.vari)

if args.corrupted == 0:
	logger('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
		test_loss, correct, len(test_loader.dataset), acc))
else:
	logger('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
		test_loss, correct, total_num_img, acc))

# call(["/bin/bash", "./layer_record/trace_command.sh"])

finish_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
