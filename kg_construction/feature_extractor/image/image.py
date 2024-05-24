import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


def load(path):
	img = np.asarray(Image.open(path).resize((224, 224), Image.BICUBIC))
	return img.astype(np.float32)


def prepare_model():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = create_feature_extractor(resnet101(pretrained=True), return_nodes={'flatten': 'feature'})
	model = model.to(device)
	for param in model.parameters():
		param.requires_grad = False
	#model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
	model.eval()
	return model


def process_data(node):
	# 加载并将形状改为 (b,c,H,W)
	img = load(node['path']) / 256
	if len(img.shape) < 3:
		#print('detected one channel image')
		img = img.reshape(img.shape+(1,))
		img = np.concatenate([img, img, img], axis=2)
	if img.shape[2] >= 4:
		img = img[:,:,:3]
	img = torch.from_numpy(img).permute(2, 0, 1)
	img = img.reshape((1,)+img.shape)
	#print(img.shape)

	# 移动到 GPU
	#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	#img = img.to(device)

	# 正则化
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	img = normalize(img)

	return img


def process_extract(img, model):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	img = img.to(device)
	result = model.forward(img)['feature']
	#print(result.shape)
	#print(result)
	return result
