import json
import torch
import librosa
import warnings
import argparse
import numpy as np
from utils import io
from audio import vggish
from image import image
from visual import visual

# to evaluate time cost
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
args = parser.parse_args()

size = args.size
rank = args.rank

io.init(size, rank)

torch.cuda.set_device(rank)


# ----- load or init checkpoint ----- #

try:
	with open(str(rank)+'checkpoint.json', 'r') as f:
		checkpoint = json.load(f)
except:
	checkpoint = {
		'stage': 0,  # current stage
		'index': 0,  # index of NEXT entry to calculate
	}

print(checkpoint)

def milestone():
	with open(str(rank)+'checkpoint.json', 'w') as f:
		f.write(json.dumps(checkpoint))
	io.milestone()


# ----- fail record ----- #

try:
	with open(str(rank)+'fail_list.json', 'r') as f:
		fail_list = json.load(f)
except:
	fail_list = []

def add_fail(node):
	if 'result' in node:
		del node['result']
	if 'frame' in node:
		del node['frame']
	if 'motion' in node:
		del node['motion']
	fail_list.append(node)
	with open(str(rank)+'fail_list.json', 'w') as f:
		f.write(json.dumps(fail_list))


# ----- audio ----- #

def process_audio():
	print('audio')

	audio_list = io.load('audio')

	print('start loading model')
	model = vggish.VGGish(urls={
		'vggish': 'audio/vggish-10086976.pth',
		'pca': 'audio/vggish_pca_params-970ea276.pth',
		})
	for param in model.parameters():
		param.requires_grad = False
	model.eval()
	print('finish loading model')


	disp_int = max(int(len(audio_list) / 50), 1)
	start = time()
	for i, e in list(enumerate(audio_list))[checkpoint['index']:]:
		try:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				samples, sr = librosa.load(e['path'], sr=None)
			# cut off audio clip that is too long (avoid exeeding GPU memory)
			# keep only the first 60 seconds
			if samples.shape[0] / sr > 60:
				samples = samples[:60*sr]
			samples = samples.astype(np.float64)
			e['result'] = model.forward(samples, sr).detach().cpu().numpy()

			io.save(e)
			#io.show_h5(e)
		except Exception as msg:
			print(msg)
			add_fail(e)

		if i % disp_int == 0:
			print('audio', i, '/', len(audio_list))
			checkpoint['index'] = i + 1
			milestone()
	end = time()
	print('in second:', (end - start) / len(audio_list))


# ----- image ----- #

def process_image():
	print('image')

	image_list = io.load('image')

	print('start loading model')
	model = image.prepare_model()
	print('finish loading model')

	disp_int = max(int(len(image_list) / 50), 1)
	start = time()
	i = checkpoint['index']
	pos = int(i / disp_int)  # help to decide when to save checkpoint
	while i < len(image_list):
		batch_data = []
		batch_node = []
		while i < len(image_list) and len(batch_data) < 32:
			try:
				batch_data.append(image.process_data(image_list[i]))
				batch_node.append(i)
			except Exception as msg:
				print(msg)
				add_fail(image_list[i])
			i += 1

		batch_data = torch.cat(batch_data, axis=0)
		result = image.process_extract(batch_data, model).detach().cpu().numpy()
		for j, node in enumerate(batch_node):
			image_list[node]['result'] = result[j]
			io.save(image_list[node])

		if int(i / disp_int) != pos:
			pos = int(i / disp_int)
			print('image', i, '/', len(image_list))
			checkpoint['index'] = i
			milestone()
	end = time()
	print('in second:', (end - start) / len(image_list))


# ----- visual ----- #

def process_visual():
	print('visual')

	visual_list = io.load('visual')

	print('start loading model')
	model = visual.prepare_model()
	print('finish loading model')

	disp_int = max(int(len(visual_list) / 50), 1)
	start = time()
	for i, e in list(enumerate(visual_list))[checkpoint['index']:]:
		try:
			visual.process(e, model)
			io.save(e)
		except Exception as msg:
			print(msg)
			add_fail(e)

		if i % disp_int == 0:
			print('visual', i, '/', len(visual_list))
			checkpoint['index'] = i + 1
			milestone()
	end = time()
	print('in second:', (end - start) / len(visual_list))


if checkpoint['stage'] == 0:
	process_image()
	checkpoint['stage'] += 1
	checkpoint['index'] = 0
	milestone()

if checkpoint['stage'] == 1:
	process_audio()
	checkpoint['stage'] += 1
	checkpoint['index'] = 0
	milestone()

if checkpoint['stage'] == 2:
	process_visual()
	checkpoint['stage'] += 1
	checkpoint['index'] = 0
	milestone()

io.end()
