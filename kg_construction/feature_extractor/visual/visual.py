import torch
import imageio
import numpy as np
from PIL import Image
from .model import resnext
from torchvision import transforms
from torchvision.models import resnet101#, resnext101_32x8d
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

# evaluate time cost
from time import time


def load(path):
	'''
	:return: [t, h, w, c] (gif has 4 channels. with alpha at index 3)
	'''
	frames = []
	for f in imageio.get_reader(path):
		f = Image.fromarray(f).resize((112, 112), Image.BICUBIC)
		f = np.array(f)
		if len(f.shape) < 3:
			f = f.reshape(f.shape+(1,))
			f = np.concatenate([f, f, f], axis=2)
		f = f[:,:,:3] / 256
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		f = normalize(torch.from_numpy(np.array(f.transpose(2, 0, 1)))).numpy()
		f = f.transpose(1, 2, 0)
		frames.append(f.reshape((1,)+f.shape))
	frames = np.concatenate(frames)
	return frames


def prepare_model():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	frame = create_feature_extractor(resnet101(pretrained=True), return_nodes={'flatten': 'feature'})
	frame = frame.to(device)
	for param in frame.parameters():
		param.requires_grad = False
	#frame = torch.nn.DataParallel(frame, device_ids=[0, 1])
	frame.eval()

	#motion = create_feature_extractor(resnext101_32x8d(pretrained=True), return_nodes={'flatten': 'feature'})
	motion = resnext.resnet101(num_classes=400, shortcut_type='B', cardinality=32,
							   sample_size=112, sample_duration=16,
							   last_fc=False)
	motion = motion.to(device)
	print(torch.cuda.current_device())
	motion = torch.nn.DataParallel(motion, device_ids=[torch.cuda.current_device()])
	model_data = torch.load('visual/model/resnext-101-kinetics.pth', map_location='cpu')
	motion.load_state_dict(model_data['state_dict'])
	for param in motion.parameters():
		param.requires_grad = False
	motion.eval()

	return {'frame': frame, 'motion': motion}


def slicing(data, num_clips=8, num_frames_per_clip=16):
	'''
	:return: num_clips * [t, c, h, w]
	'''
	total_frames = data.shape[0]
	#img_size = (224, 224)

	clips = list()

	for i in np.linspace(0, total_frames, num_clips + 2, dtype=np.int32)[1:num_clips + 1]:
		# locate the frame range
		clip_start = int(i) - int(num_frames_per_clip / 2)
		clip_end = int(i) + int(num_frames_per_clip / 2)
		if clip_start < 0:
			clip_start = 0
		if clip_end > total_frames:
			clip_end = total_frames - 1
		clip = data[clip_start:clip_end]
		# add necessary padding
		if clip_start == 0:
			shortage = num_frames_per_clip - (clip_end - clip_start)
			added_frames = []
			for _ in range(shortage):
				added_frames.append(np.expand_dims(data[clip_start], axis=0))
			if len(added_frames) > 0:
				added_frames = np.concatenate(added_frames, axis=0)
				clip = np.concatenate((added_frames, clip), axis=0)
		if clip_end == (total_frames - 1):
			shortage = num_frames_per_clip - (clip_end - clip_start)
			added_frames = []
			for _ in range(shortage):
				added_frames.append(np.expand_dims(data[clip_end], axis=0))
			if len(added_frames) > 0:
				added_frames = np.concatenate(added_frames, axis=0)
				clip = np.concatenate((clip, added_frames), axis=0)

		# sample each frame within this slice
		new_clip = []
		for j in range(num_frames_per_clip):
			frame_data = clip[j]
			frame_data = frame_data.transpose(2, 0, 1)
			new_clip.append(frame_data)
		new_clip = np.asarray(new_clip)  # (num_frames, channels, width, height)
		clips.append(new_clip)

	return clips


def process_motion(data, model):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	vi = torch.FloatTensor(np.asarray(data)).to(device)  # [num_clips, t, c, h, w]
	vi = vi.permute(0, 2, 1, 3, 4)  # [num_clips, c, t, h, w] (because we are doing 3D conv)
	#print(vi.shape)

	vi = model(vi)
	#print(vi.shape)

	return vi


def process_frame(data, model):
	''' result is transformed to numpy array already '''
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# 8 groups
	#vi = []
	#for clip in data:
	#	clip_torch = torch.FloatTensor(clip).to(device)
	#	with torch.no_grad():
	#		vi.append(model(clip_torch)['feature'].cpu().numpy())
	#vi = np.asarray(vi)

	# 2 groups
	#clips = torch.FloatTensor(np.asarray(data)).to(device)
	#vi = []
	#for i in range(0, 8, 4):
	#	clip = clips[i:i+4,:,:,:,:].reshape((4*16), 3, 112, 112)
	#	vi.append(model(clip)['feature'].reshape((4, 16, 2048)).cpu().numpy())
	#vi = np.concatenate(vi)

	# once for all
	vi = torch.FloatTensor(np.asarray(data)).to(device)
	vi = vi.reshape((8*16), 3, 112, 112)
	vi = model(vi)['feature'].cpu().numpy()
	vi = vi.reshape((8, 16, 2048))

	#print(vi.shape)
	return vi


def process(node, model):
	vi = load(node['path'])
	vi = slicing(vi)
	node['motion'] = process_motion(vi, model['motion'])
	node['frame'] = process_frame(vi, model['frame'])
