import tensorflow as tf
import os

mapping_size = 100
relation_structural_embeddings_size = 40
entity_structural_embeddings_size = 40
#entity_multimodal_embeddings_size = 300
use_text = True
entity_text_embeddings_size = 300
use_image = True
entity_image_embeddings_size = 2048
#relation_image_embeddings_size = 2048
use_video = True
entity_video_embeddings_size = 8 * 2048
#relation_audio_embeddings_size = 8 * 16 * 2048# + 8 * 2048
use_audio = True
audio_duration = 50
audio_per_frame_size = 128
entity_audio_embeddings_size = audio_per_frame_size * audio_duration
#relation_video_embeddings_size = 128 * audio_duration

use_rel_mm = True

#nr_neuron_dense_layer_sum = 100
nr_neuron_dense_layer_1 = 2048
nr_neuron_dense_layer_2 = 1024
dropout_ratio = 0.0
margin = 10
training_epochs = 1000
batch_size = 100
display_step = 1
activation_function = tf.nn.tanh
initial_learning_rate = 0.001
head_mult = [1 for _ in range(5)] + [1, 1, 1, 1, 1]
tail_mult = [1 for _ in range(5)] + [1, 1, 1, 0, 0]


# Loading the data

all_triples_file =   "/path/to/all.txt" #"
train_triples_file = "/path/to/train.txt" #
test_triples_file =  "/path/to/test.txt"
valid_triples_file =  "/path/to/valid.txt"

entity_full_info = '/path/to/entities.json'
relation_full_info = '/path/to/relations.json'

structure_embedding_file = '/path/to/structure.hdf5'
entity2id = '/path/to/entity2id.json'
relation2id = '/path/to/relation2id.json'
text_embedding_file = '/path/to/my_embedding.hdf5'
multimodal_embedding_file = '/path/to/all.hdf5'



#model_id = "FBIMG_HMS_MM128_dropout0_m10_tanh_mapped_1_layer_02" #_mm_loss_10m" #"HMS_standard_vgg128_noreg" #"HMS_standard_full_mapping_elu_300_100"
model_id = 'tiva'

checkpoint_best_valid_dir = "weights/best_"+model_id+"/"
checkpoint_current_dir ="weights/current_"+model_id+"/"
results_dir = "results/results_"+model_id+"/"

if not os.path.exists(checkpoint_best_valid_dir):
    os.makedirs(checkpoint_best_valid_dir)

if not os.path.exists(checkpoint_current_dir):
    os.makedirs(checkpoint_current_dir)


if not os.path.exists(results_dir):
    os.makedirs(results_dir)


model_current_weights_file = checkpoint_current_dir + model_id + "_current"
current_model_meta_file = checkpoint_current_dir + model_id + "_current.meta"

model_weights_best_valid_file = checkpoint_best_valid_dir + model_id + "_best_hits"
best_valid_model_meta_file = checkpoint_best_valid_dir + model_id + "_best_hits.meta"


result_file = results_dir+model_id+"_results.txt"
log_file = results_dir+model_id+"_log.txt"
loss_file = results_dir+model_id+"_loss.json"

