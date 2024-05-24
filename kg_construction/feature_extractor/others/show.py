#/DATA/DATANAS1/hrz/videoqa/hcrn-vggsound/data/tgif-qa/action

import h5py

path = '/DATA/DATANAS1/hrz/videoqa/hcrn-vggsound/data/tgif-qa/action/tgif-qa_action_motion_feat.h5'

with h5py.File(path, 'r') as f:
	print(f['resnext_features'])
