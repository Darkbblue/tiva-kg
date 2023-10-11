# Specify the parameters for the test

entity_text_embeddings_size = 300
entity_image_embeddings_size = 2048
#relation_image_embeddings_size = 2048
entity_video_embeddings_size = 8 * 2048
#relation_audio_embeddings_size = 8 * 16 * 2048# + 8 * 2048
audio_duration = 50
audio_per_frame_size = 128
entity_audio_embeddings_size = audio_per_frame_size * audio_duration


entity_full_info = '/path/to/entities.json'
relation_full_info = '/path/to/relations.json'

structure_embedding_file = '/path/to/structure.hdf5'
entity2id = '/path/to/entity2id.json'
relation2id = '/path/to/relation2id.json'
text_embedding_file = '/path/to/my_embedding.hdf5'
multimodal_embedding_file = '/path/to/all.hdf5'

all_triples_file =   "/path/to/all.txt" #"
train_triples_file = "/path/to/train.txt" #
test_triples_file =  "/path/to/test.txt"
valid_triples_file =  "/path/to/valid.txt"


model_id = "tiva"

strict_relation = False

# where to load the weights for the model
checkpoint_best_valid_dir = "weights/best_"+model_id+"/"
model_weights_best_valid_file = checkpoint_best_valid_dir + model_id + "_best_hits"
best_valid_model_meta_file = checkpoint_best_valid_dir + model_id + "_best_hits.meta"

checkpoint_current_dir ="weights/current_"+model_id+"/"
model_current_weights_file = checkpoint_current_dir + model_id + "_current"
current_model_meta_file = checkpoint_current_dir + model_id + "_current.meta"

# Results location
results_dir = "results/results_"+model_id+"/"
result_file = results_dir+model_id
if strict_relation:
	result_file = result_file + "_strict_relation"
result_file = result_file + "_results.txt"

