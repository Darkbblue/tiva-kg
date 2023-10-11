import os

import numpy as np
import tensorflow as tf

import parameters as param
import util as u

import json

#tf.set_random_seed(1234)
#np.random.seed(7)

logs_path = "log"
# .... Loading the data ....
print("load all triples")
#relation_embeddings = u.load_binary_file(param.relation_structural_embeddings_file)
#entity_embeddings_txt = u.load_binary_file(param.entity_structural_embeddings_file)
#entity_embeddings_img = u.load_binary_file(param.entity_multimodal_embeddings_file)
# structure embedding, and 2id.json required to correctly load structure embedding
structure_embedding = u.load_embedding(param.structure_embedding_file)
entity2id = u.load_json(param.entity2id)
relation2id = u.load_json(param.relation2id)
# text
text_embedding = u.load_embedding(param.text_embedding_file)
# multimodal embedding and the json required to load it
multimodal_embedding = u.load_embedding(param.multimodal_embedding_file)
entity_full_info = u.load_json(param.entity_full_info)
relation_full_info = u.load_json(param.relation_full_info)

# organize embedding data all in one dict
embeddings = {
    'structure': structure_embedding,
    'entity2id': entity2id,
    'relation2id': relation2id,
    'text': text_embedding,
    'multimodal': multimodal_embedding,
    'entity': entity_full_info,
    'relation': relation_full_info
}

all_train_test_valid_triples, entity_list = u.load_training_triples(param.all_triples_file)
triples_set = [t[0] + "_" + t[1] + "_" + t[2] for t in all_train_test_valid_triples]
triples_set = set(triples_set)
_, train_entity_list = u.load_training_triples(param.train_triples_file)
_, valid_entity_list = u.load_training_triples(param.valid_triples_file)
#entity_list_filtered = []
#for e in entity_list:
#    if e in entity_embeddings_txt:
#        entity_list_filtered.append(e)
#entity_list = entity_list_filtered
print("#entities", len(entity_list), "#total triples", len(all_train_test_valid_triples))

# global dict to store all embedding vectors according to the key (of entity or relation)
all_entities = {}
all_relations = {}

training_data = u.load_full_data(param.train_triples_file, embeddings, all_entities, all_relations)

#training_data= training_data[:1000]
#training_data = training_data[:10000]
print("#training data", len(training_data))

valid_data = u.load_full_data(param.valid_triples_file, embeddings, all_entities, all_relations)
#valid_data = valid_data[:len(valid_data)//3]

print("valid_data",len(valid_data))

def max_norm_regulizer(threshold,axes=1,name="max_norm",collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights,clip_norm=threshold,axes=axes)
        clip_weights = tf.assign(weights,clipped,name=name)
        tf.add_to_collection(collection,clip_weights)
        return None
    return max_norm

max_norm_reg = max_norm_regulizer(threshold=1.0)

def my_dense(x, nr_hidden, scope, activation_fn=param.activation_function,reuse=None):
    with tf.variable_scope(scope):
        h = tf.contrib.layers.fully_connected(x, nr_hidden,
                                              activation_fn=activation_fn,
                                              reuse=reuse,
                                              scope=scope#, weights_regularizer= max_norm_reg
                                              )

        return h



# ........... Creating the model
with tf.name_scope('input'):
    text_padding = tf.Variable(tf.random.uniform([1, param.entity_text_embeddings_size]), trainable=True)
    image_padding = tf.Variable(tf.random.uniform([1, param.entity_image_embeddings_size]), trainable=True)
    video_padding = tf.Variable(tf.random.uniform([1, param.entity_video_embeddings_size]), trainable=True)
    audio_padding = tf.Variable(tf.random.uniform([1, param.entity_audio_embeddings_size]), trainable=True)

    head_mult = param.head_mult
    tail_mult = param.tail_mult


    r_input = tf.placeholder(dtype=tf.float32, shape=[None, param.relation_structural_embeddings_size],name="r_input")

    r_image_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_image_embeddings_size], name='r_image_input')
    r_image_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='r_image_padding')
    r_image_masked = r_image_input + r_image_padding * image_padding

    r_video_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_video_embeddings_size], name='r_video_input')
    r_video_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='r_video_padding')
    r_video_masked = r_video_input + r_video_padding * video_padding

    r_audio_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_audio_embeddings_size], name='r_audio_input')
    r_audio_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='r_audio_padding')
    r_audio_masked = r_audio_input + r_audio_padding * audio_padding


    h_pos_structure_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size], name="h_pos_structure_input")
    h_pos_text_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_text_embeddings_size],name="h_pos_text_input")
    h_pos_text_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="h_pos_text_padding")
    h_pos_text_masked = h_pos_text_input + h_pos_text_padding * text_padding

    h_pos_image_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_image_embeddings_size],name="h_pos_image_input")
    h_pos_image_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="h_pos_image_padding")
    h_pos_image_masked = h_pos_image_input + h_pos_image_padding * image_padding

    h_pos_video_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_video_embeddings_size],name="h_pos_video_input")
    h_pos_video_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="h_pos_video_padding")
    h_pos_video_masked = h_pos_video_input + h_pos_video_padding * video_padding

    h_pos_audio_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_audio_embeddings_size],name="h_pos_audio_input")
    h_pos_audio_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="h_pos_audio_padding")
    h_pos_audio_masked = h_pos_audio_input + h_pos_audio_padding * audio_padding


    t_pos_structure_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size], name="t_pos_structure_input")
    t_pos_text_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_text_embeddings_size], name="t_pos_text_input")
    t_pos_text_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="t_pos_text_padding")
    t_pos_text_masked = t_pos_text_input + t_pos_text_padding * text_padding

    t_pos_image_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_image_embeddings_size], name="t_pos_image_input")
    t_pos_image_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="t_pos_image_padding")
    t_pos_image_masked = t_pos_image_input + t_pos_image_padding * image_padding

    t_pos_video_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_video_embeddings_size], name="t_pos_video_input")
    t_pos_video_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="t_pos_video_padding")
    t_pos_video_masked = t_pos_video_input + t_pos_video_padding * video_padding

    t_pos_audio_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_audio_embeddings_size], name="t_pos_audio_input")
    t_pos_audio_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="t_pos_audio_padding")
    t_pos_audio_masked = t_pos_audio_input + t_pos_audio_padding * audio_padding


    h_neg_structure_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size], name="h_neg_structure_input")
    h_neg_text_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_text_embeddings_size], name="h_neg_text_input")
    h_neg_text_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="h_neg_text_padding")
    h_neg_text_masked = h_neg_text_input + h_neg_text_padding * text_padding

    h_neg_image_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_image_embeddings_size], name="h_neg_image_input")
    h_neg_image_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="h_neg_image_padding")
    h_neg_image_masked = h_neg_image_input + h_neg_image_padding * image_padding

    h_neg_video_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_video_embeddings_size], name="h_neg_video_input")
    h_neg_video_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="h_neg_video_padding")
    h_neg_video_masked = h_neg_video_input + h_neg_video_padding * video_padding

    h_neg_audio_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_audio_embeddings_size], name="h_neg_audio_input")
    h_neg_audio_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="h_neg_audio_padding")
    h_neg_audio_masked = h_neg_audio_input + h_neg_audio_padding * audio_padding


    t_neg_structure_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size],   name="t_neg_structure_input")
    t_neg_text_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_text_embeddings_size], name="t_neg_text_input")
    t_neg_text_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="t_neg_text_padding")
    t_neg_text_masked = t_neg_text_input + t_neg_text_padding * text_padding

    t_neg_image_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_image_embeddings_size], name="t_neg_image_input")
    t_neg_image_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="t_neg_image_padding")
    t_neg_image_masked = t_neg_image_input + t_neg_image_padding * image_padding

    t_neg_video_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_video_embeddings_size], name="t_neg_video_input")
    t_neg_video_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="t_neg_video_padding")
    t_neg_video_masked = t_neg_video_input + t_neg_video_padding * video_padding

    t_neg_audio_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_audio_embeddings_size], name="t_neg_audio_input")
    t_neg_audio_padding = tf.placeholder(dtype=tf.float32, shape=[None, 1],name="t_neg_audio_padding")
    t_neg_audio_masked = t_neg_audio_input + t_neg_audio_padding * audio_padding


    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

with tf.name_scope('head_relation'):
    # structure
    r_mapped = my_dense(r_input, param.mapping_size, activation_fn=param.activation_function, scope="txt_proj", reuse=None)
    r_mapped = tf.nn.dropout(r_mapped, keep_prob)

    h_pos_structure_mapped = my_dense(h_pos_structure_input, param.mapping_size, activation_fn=param.activation_function,  scope="txt_proj", reuse=True)
    h_pos_structure_mapped = tf.nn.dropout(h_pos_structure_mapped, keep_prob)

    h_neg_structure_mapped = my_dense(h_neg_structure_input, param.mapping_size, activation_fn=param.activation_function,  scope="txt_proj", reuse=True)
    h_neg_structure_mapped = tf.nn.dropout(h_neg_structure_mapped, keep_prob)

    t_pos_structure_mapped = my_dense(t_pos_structure_input, param.mapping_size, activation_fn=param.activation_function,   scope="txt_proj", reuse=True)
    t_pos_structure_mapped = tf.nn.dropout(t_pos_structure_mapped, keep_prob)

    t_neg_structure_mapped = my_dense(t_neg_structure_input, param.mapping_size, activation_fn=param.activation_function,   scope="txt_proj", reuse=True)
    t_neg_structure_mapped = tf.nn.dropout(t_neg_structure_mapped, keep_prob)

    ltsm_cell = tf.nn.rnn_cell.BasicLSTMCell(param.audio_per_frame_size)

    def concat_mm(text, image, video, audio, is_rel=False):
        mm = tf.ones([tf.shape(r_mapped)[0], 0])
        # print(mm)
        if not is_rel and param.use_text:
            mm = tf.concat([mm, text], axis=1)
        if param.use_image:
            mm = tf.concat([mm, image], axis=1)
        if param.use_video:
            mm = tf.concat([mm, video], axis=1)
        if param.use_audio:
            audio = tf.reshape(audio, [-1, param.audio_duration, param.audio_per_frame_size])
            audio = tf.nn.dynamic_rnn(ltsm_cell, audio, dtype=tf.float32)[0][:,-1,:]
            mm = tf.concat([mm, audio], axis=1)
        return mm

    # Multi-modal

    r_mm = concat_mm(None, r_image_masked, r_video_masked, r_audio_masked, is_rel=True)
    r_mm_mapped = my_dense(r_mm, param.mapping_size, activation_fn=param.activation_function, scope="rel_mm_proj", reuse=None)
    r_mm_mapped = tf.nn.dropout(r_mm_mapped, keep_prob)

    h_pos_mm = concat_mm(h_pos_text_masked, h_pos_image_masked, h_pos_video_masked, h_pos_audio_masked)
    h_pos_mm_mapped = my_dense(h_pos_mm, param.mapping_size, activation_fn=param.activation_function, scope="entity_mm_proj", reuse=None)
    h_pos_mm_mapped = tf.nn.dropout(h_pos_mm_mapped, keep_prob)

    h_neg_mm = concat_mm(h_neg_text_masked, h_neg_image_masked, h_neg_video_masked, h_neg_audio_masked)
    h_neg_mm_mapped = my_dense(h_neg_mm, param.mapping_size, activation_fn=param.activation_function, scope="entity_mm_proj", reuse=True)
    h_neg_mm_mapped = tf.nn.dropout(h_neg_mm_mapped, keep_prob)

    t_pos_mm = concat_mm(t_pos_text_masked, t_pos_image_masked, t_pos_video_masked, t_pos_audio_masked)
    t_pos_mm_mapped = my_dense(t_pos_mm, param.mapping_size, activation_fn=param.activation_function, scope="entity_mm_proj", reuse=True)
    t_pos_mm_mapped = tf.nn.dropout(t_pos_mm_mapped, keep_prob)

    t_neg_mm = concat_mm(t_neg_text_masked, t_neg_image_masked, t_neg_video_masked, t_neg_audio_masked)
    t_neg_mm_mapped = my_dense(t_neg_mm, param.mapping_size, activation_fn=param.activation_function, scope="entity_mm_proj", reuse=True)
    t_neg_mm_mapped = tf.nn.dropout(t_neg_mm_mapped, keep_prob)




with tf.name_scope('cosine'):

    # Head model
    energy_ss_pos = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_pos_structure_mapped), 1, keep_dims=True, name="pos_s_s")
    energy_ss_neg = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_neg_structure_mapped), 1, keep_dims=True, name="neg_s_s")
    energy_ss_pos = energy_ss_pos * head_mult[0]
    energy_ss_neg = energy_ss_neg * head_mult[0]

    energy_is_pos = tf.reduce_sum(abs(h_pos_mm_mapped + r_mapped - t_pos_structure_mapped), 1, keep_dims=True, name="pos_i_i")
    energy_is_neg = tf.reduce_sum(abs(h_pos_mm_mapped + r_mapped - t_neg_structure_mapped), 1, keep_dims=True, name="neg_i_i")
    energy_is_pos = energy_is_pos * head_mult[1]
    energy_is_neg = energy_is_neg * head_mult[1]

    energy_si_pos = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_pos_mm_mapped), 1, keep_dims=True, name="pos_s_i")
    energy_si_neg = tf.reduce_sum(abs(h_pos_structure_mapped + r_mapped - t_neg_mm_mapped), 1, keep_dims=True, name="neg_s_i")
    energy_si_pos = energy_si_pos * head_mult[2]
    energy_si_neg = energy_si_neg * head_mult[2]

    energy_ii_pos = tf.reduce_sum(abs(h_pos_mm_mapped + r_mapped - t_pos_mm_mapped), 1, keep_dims=True, name="pos_i_i")
    energy_ii_neg = tf.reduce_sum(abs(h_pos_mm_mapped + r_mapped - t_neg_mm_mapped), 1, keep_dims=True, name="neg_i_i")
    energy_ii_pos = energy_ii_pos * head_mult[3]
    energy_ii_neg = energy_ii_neg * head_mult[3]

    energy_concat_pos = tf.reduce_sum(abs((h_pos_structure_mapped + h_pos_mm_mapped) + r_mapped - (t_pos_structure_mapped + t_pos_mm_mapped)), 1, keep_dims=True, name="energy_concat_pos")
    energy_concat_neg = tf.reduce_sum(abs((h_pos_structure_mapped + h_pos_mm_mapped) + r_mapped - (t_neg_structure_mapped + t_neg_mm_mapped)), 1, keep_dims=True, name="energy_concat_neg")
    energy_concat_pos = energy_concat_pos * head_mult[4]
    energy_concat_neg = energy_concat_neg * head_mult[4]

    if param.use_rel_mm:
        # mm energy - h, r, t all use mm feature
        energy_iii_pos = tf.reduce_sum(abs(h_pos_mm_mapped + r_mm_mapped - t_pos_mm_mapped), 1, keep_dims=True, name="pos_i_i_i")
        energy_iii_neg = tf.reduce_sum(abs(h_pos_mm_mapped + r_mm_mapped - t_neg_mm_mapped), 1, keep_dims=True, name="neg_i_i_i")
        energy_iii_pos = energy_iii_pos * head_mult[5]
        energy_iii_neg = energy_iii_neg * head_mult[5]

        # mm energy - concat
        energy_concat2_pos = tf.reduce_sum(abs((h_pos_structure_mapped + h_pos_mm_mapped) + r_mm_mapped - (t_pos_structure_mapped + t_pos_mm_mapped)), 1, keep_dims=True, name="energy_concat2_pos")
        energy_concat2_neg = tf.reduce_sum(abs((h_pos_structure_mapped + h_pos_mm_mapped) + r_mm_mapped - (t_neg_structure_mapped + t_neg_mm_mapped)), 1, keep_dims=True, name="energy_concat2_neg")
        energy_concat2_pos = energy_concat2_pos * head_mult[6]
        energy_concat2_neg = energy_concat2_neg * head_mult[6]

        # sis
        energy_sis_pos = tf.reduce_sum(abs(h_pos_structure_mapped + r_mm_mapped - t_pos_structure_mapped), 1, keep_dims=True, name="pos_s_i_s")
        energy_sis_neg = tf.reduce_sum(abs(h_pos_structure_mapped + r_mm_mapped - t_neg_structure_mapped), 1, keep_dims=True, name="neg_s_i_s")
        energy_sis_pos = energy_sis_pos * head_mult[7]
        energy_sis_neg = energy_sis_neg * head_mult[7]

        # mms
        energy_iis_pos = tf.reduce_sum(abs(h_pos_mm_mapped + r_mm_mapped - t_pos_structure_mapped), 1, keep_dims=True, name="pos_i_i_s")
        energy_iis_neg = tf.reduce_sum(abs(h_pos_mm_mapped + r_mm_mapped - t_neg_structure_mapped), 1, keep_dims=True, name="neg_i_i_s")
        energy_iis_pos = energy_iis_pos * head_mult[8]
        energy_iis_neg = energy_iis_neg * head_mult[8]

        # smm
        energy_sii_pos = tf.reduce_sum(abs(h_pos_structure_mapped + r_mm_mapped - t_pos_mm_mapped), 1, keep_dims=True, name="pos_s_i_i")
        energy_sii_neg = tf.reduce_sum(abs(h_pos_structure_mapped + r_mm_mapped - t_neg_mm_mapped), 1, keep_dims=True, name="neg_s_i_i")
        energy_sii_pos = energy_sii_pos * head_mult[9]
        energy_sii_neg = energy_sii_neg * head_mult[9]

        h_r_t_pos = tf.reduce_sum([energy_ss_pos, energy_is_pos, energy_si_pos, energy_ii_pos, energy_concat_pos, energy_iii_pos, energy_concat2_pos, energy_sis_pos, energy_iis_pos, energy_sii_pos], 0,   name="h_r_t_pos")
        h_r_t_neg = tf.reduce_sum([energy_ss_neg, energy_is_neg, energy_si_neg, energy_ii_neg, energy_concat_neg, energy_iii_neg, energy_concat2_neg, energy_sis_neg, energy_iis_neg, energy_sii_neg], 0, name="h_r_t_neg")
    else:
        h_r_t_pos = tf.reduce_sum([energy_ss_pos, energy_is_pos, energy_si_pos, energy_ii_pos, energy_concat_pos], 0,   name="h_r_t_pos")
        h_r_t_neg = tf.reduce_sum([energy_ss_neg, energy_is_neg, energy_si_neg, energy_ii_neg, energy_concat_neg], 0, name="h_r_t_neg")


    # Tail model

    score_t_t_pos = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_pos_structure_mapped), 1, keep_dims=True, name="pos_s_s")
    score_t_t_neg = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_neg_structure_mapped), 1, keep_dims=True, name="neg_s_s")
    score_t_t_pos = score_t_t_pos * tail_mult[0]
    score_t_t_neg = score_t_t_neg * tail_mult[0]

    score_i_t_pos = tf.reduce_sum(abs(t_pos_mm_mapped - r_mapped - h_pos_structure_mapped), 1, keep_dims=True, name="pos_i_s")
    score_i_t_neg = tf.reduce_sum(abs(t_pos_mm_mapped - r_mapped - h_neg_structure_mapped), 1, keep_dims=True, name="neg_i_s")
    score_i_t_pos = score_i_t_pos * tail_mult[1]
    score_i_t_neg = score_i_t_neg * tail_mult[1]

    score_t_i_pos = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_pos_mm_mapped), 1, keep_dims=True, name="pos_s_i")
    score_t_i_neg = tf.reduce_sum(abs(t_pos_structure_mapped - r_mapped - h_neg_mm_mapped), 1, keep_dims=True, name="neg_s_i")
    score_t_i_pos = score_t_i_pos * tail_mult[2]
    score_i_t_neg = score_i_t_neg * tail_mult[2]

    score_i_i_pos = tf.reduce_sum(abs(t_pos_mm_mapped - r_mapped - h_pos_mm_mapped), 1, keep_dims=True, name="pos_i_i")
    score_i_i_neg = tf.reduce_sum(abs(t_pos_mm_mapped - r_mapped - h_neg_mm_mapped), 1, keep_dims=True, name="neg_i_i")
    score_i_i_pos = score_i_i_pos * tail_mult[3]
    score_i_i_neg = score_i_i_neg * tail_mult[3]

    energy_concat_pos_tail = tf.reduce_sum(abs((t_pos_structure_mapped + t_pos_mm_mapped) - r_mapped - (h_pos_structure_mapped + h_pos_mm_mapped)), 1, keep_dims=True, name="energy_concat_pos_tail")
    energy_concat_neg_tail = tf.reduce_sum(abs((t_pos_structure_mapped + t_pos_mm_mapped) - r_mapped - (h_neg_structure_mapped + h_neg_mm_mapped)), 1, keep_dims=True, name="energy_concat_neg_tail")
    energy_concat_pos_tail = energy_concat_pos_tail * tail_mult[4]
    energy_concat_neg_tail = energy_concat_neg_tail * tail_mult[4]

    if param.use_rel_mm:
        # mm energy - h, r, t all use mm feature
        score_iii_pos = tf.reduce_sum(abs(t_pos_mm_mapped - r_mm_mapped - h_pos_mm_mapped), 1, keep_dims=True, name="pos_i_i_i")
        score_iii_neg = tf.reduce_sum(abs(t_pos_mm_mapped - r_mm_mapped - h_neg_mm_mapped), 1, keep_dims=True, name="neg_i_i_i")
        score_iii_pos = score_iii_pos * tail_mult[5]
        score_iii_neg = score_iii_neg * tail_mult[5]

        # mm energy - concat
        energy_concat2_pos_tail = tf.reduce_sum(abs((t_pos_structure_mapped + t_pos_mm_mapped) - r_mm_mapped - (h_pos_structure_mapped + h_pos_mm_mapped)), 1, keep_dims=True, name="energy_concat2_pos")
        energy_concat2_neg_tail = tf.reduce_sum(abs((t_pos_structure_mapped + t_pos_mm_mapped) - r_mm_mapped - (h_neg_structure_mapped + h_neg_mm_mapped)), 1, keep_dims=True, name="energy_concat2_neg")
        energy_concat2_pos_tail = energy_concat2_pos_tail * tail_mult[6]
        energy_concat2_neg_tail = energy_concat2_neg_tail * tail_mult[6]

        # sis
        score_sis_pos = tf.reduce_sum(abs(t_pos_structure_mapped - r_mm_mapped - h_pos_structure_mapped), 1, keep_dims=True, name="pos_s_i_s")
        score_sis_neg = tf.reduce_sum(abs(t_pos_structure_mapped - r_mm_mapped - h_neg_structure_mapped), 1, keep_dims=True, name="neg_s_i_s")
        score_sis_pos = score_sis_pos * tail_mult[7]
        score_sis_neg = score_sis_neg * tail_mult[7]

        # iis
        score_iis_pos = tf.reduce_sum(abs(t_pos_mm_mapped - r_mm_mapped - h_pos_structure_mapped), 1, keep_dims=True, name="pos_i_i_s")
        score_iis_neg = tf.reduce_sum(abs(t_pos_mm_mapped - r_mm_mapped - h_neg_structure_mapped), 1, keep_dims=True, name="neg_i_i_s")
        score_iis_pos = score_iis_pos * tail_mult[8]
        score_iis_neg = score_iis_neg * tail_mult[8]

        # sii
        score_sii_pos = tf.reduce_sum(abs(t_pos_structure_mapped - r_mm_mapped - h_pos_mm_mapped), 1, keep_dims=True, name="pos_s_i_i")
        score_sii_neg = tf.reduce_sum(abs(t_pos_structure_mapped - r_mm_mapped - h_neg_mm_mapped), 1, keep_dims=True, name="neg_s_i_i")
        score_sii_pos = score_sii_pos * tail_mult[9]
        score_sii_neg = score_sii_neg * tail_mult[9]

        t_r_h_pos = tf.reduce_sum([score_t_t_pos, score_i_t_pos, score_t_i_pos, score_i_i_pos,energy_concat_pos_tail, score_iii_pos, energy_concat2_pos_tail, score_sis_pos, score_iis_pos, score_sii_pos], 0, name="t_r_h_pos")
        t_r_h_neg = tf.reduce_sum([score_t_t_neg, score_i_t_neg, score_t_i_neg, score_i_i_neg,energy_concat_neg_tail, score_iii_neg, energy_concat2_neg_tail, score_sis_neg, score_iis_neg, score_sii_neg], 0, name="t_r_h_neg")
    else:
        t_r_h_pos = tf.reduce_sum([score_t_t_pos, score_i_t_pos, score_t_i_pos, score_i_i_pos,energy_concat_pos_tail], 0, name="t_r_h_pos")
        t_r_h_neg = tf.reduce_sum([score_t_t_neg, score_i_t_neg, score_t_i_neg, score_i_i_neg,energy_concat_neg_tail], 0, name="t_r_h_neg")


    kbc_loss1 = tf.maximum(0., param.margin - h_r_t_neg + h_r_t_pos)
    kbc_loss2 = tf.maximum(0., param.margin - t_r_h_neg + t_r_h_pos)

    kbc_loss = kbc_loss1 + kbc_loss2

    tf.summary.histogram("loss", kbc_loss)

#epsilon= 0.1
optimizer = tf.train.AdamOptimizer(param.initial_learning_rate).minimize(kbc_loss)

summary_op = tf.summary.merge_all()

#..... start the training
saver = tf.train.Saver()
log_file = open(param.log_file,"w")

log_file.write("relation_input_size = " + str(param.relation_structural_embeddings_size)+ "\n")
log_file.write("entity_input_size = " + str(param.entity_structural_embeddings_size) + "\n")
log_file.write("nr_neuron_dense_layer1 = " + str(param.nr_neuron_dense_layer_1) +"\n")
log_file.write("nr_neuron_dense_layer2 = " + str(param.nr_neuron_dense_layer_2) +"\n")
log_file.write("dropout_ratio = " + str(param.dropout_ratio) +"\n")
log_file.write("margin = " + str(param.margin) +"\n")
log_file.write("training_epochs = " + str(param.training_epochs) +"\n")
log_file.write("batch_size = " + str(param.batch_size) +"\n")
log_file.write("activation_function = " + str(param.activation_function) +"\n")
log_file.write("initial_learning_rate = " + str(param.initial_learning_rate) +"\n")

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

#np.random.shuffle(valid_data)

head_valid, rel_valid, tail_valid, head_neg_valid, tail_neg_valid = \
    u.get_batch_with_neg_heads_and_neg_tails_multimodal(valid_data,
                                                        triples_set,
                                                        valid_entity_list,
                                                        0, len(valid_data),
                                                        all_entities,
                                                        all_relations)
#clip_all_weights = tf.get_collection("max_norm")

loss_log = []

with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())

    if os.path.isfile(param.best_valid_model_meta_file):
        print("restore the weights",param.checkpoint_best_valid_dir)
        saver = tf.train.import_meta_graph(param.best_valid_model_meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(param.checkpoint_best_valid_dir))
    else:
        print("no weights to load :(")


    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    initial_valid_loss = 100


    for epoch in range(param.training_epochs):

        np.random.shuffle(training_data)

       # training_data2 = training_data[:len(training_data)//3]
        training_loss = 0.
        total_batch = len(training_data) // param.batch_size
        if len(training_data) % param.batch_size != 0:
            total_batch += 1

        for i in range(total_batch):

            batch_loss = 0
            start = i * param.batch_size
            end = (i + 1) * param.batch_size

            head_train, rel_train, tail_train, head_neg_train, tail_neg_train = u.get_batch_with_neg_heads_and_neg_tails_multimodal(
                training_data, triples_set, train_entity_list, start,
                end, all_entities, all_relations)

            _, summary, loss, o11, o12, o13, o21, o22, o23, o32 = sess.run(
                [optimizer, summary_op, kbc_loss, kbc_loss1, h_r_t_pos, h_r_t_neg, kbc_loss2, t_r_h_pos, t_r_h_neg, energy_iii_pos],
                feed_dict={r_input: rel_train['structure'],
                           r_image_input: rel_train['image'],
                           r_image_padding: rel_train['padding_image'],
                           r_video_input: rel_train['video'],
                           r_video_padding: rel_train['padding_video'],
                           r_audio_input: rel_train['audio'],
                           r_audio_padding: rel_train['padding_audio'],

                           h_pos_structure_input: head_train['structure'],
                           h_pos_text_input: head_train['text'],
                           h_pos_text_padding: head_train['padding_text'],
                           h_pos_image_input: head_train['image'],
                           h_pos_image_padding: head_train['padding_image'],
                           h_pos_video_input: head_train['video'],
                           h_pos_video_padding: head_train['padding_video'],
                           h_pos_audio_input: head_train['audio'],
                           h_pos_audio_padding: head_train['padding_audio'],

                           t_pos_structure_input: tail_train['structure'],
                           t_pos_text_input: tail_train['text'],
                           t_pos_text_padding: tail_train['padding_text'],
                           t_pos_image_input: tail_train['image'],
                           t_pos_image_padding: tail_train['padding_image'],
                           t_pos_video_input: tail_train['video'],
                           t_pos_video_padding: tail_train['padding_video'],
                           t_pos_audio_input: tail_train['audio'],
                           t_pos_audio_padding: tail_train['padding_audio'],

                           t_neg_structure_input: tail_neg_train['structure'],
                           t_neg_text_input: tail_neg_train['text'],
                           t_neg_text_padding: tail_neg_train['padding_text'],
                           t_neg_image_input: tail_neg_train['image'],
                           t_neg_image_padding: tail_neg_train['padding_image'],
                           t_neg_video_input: tail_neg_train['video'],
                           t_neg_video_padding: tail_neg_train['padding_video'],
                           t_neg_audio_input: tail_neg_train['audio'],
                           t_neg_audio_padding: tail_neg_train['padding_audio'],

                           h_neg_structure_input: head_neg_train['structure'],
                           h_neg_text_input: head_neg_train['text'],
                           h_neg_text_padding: head_neg_train['padding_text'],
                           h_neg_image_input: head_neg_train['image'],
                           h_neg_image_padding: head_neg_train['padding_image'],
                           h_neg_video_input: head_neg_train['video'],
                           h_neg_video_padding: head_neg_train['padding_video'],
                           h_neg_audio_input: head_neg_train['audio'],
                           h_neg_audio_padding: head_neg_train['padding_audio'],

                           keep_prob: 1 - param.dropout_ratio#,
                           #learning_rate : param.initial_learning_rate
                           })
            #sess.run(clip_all_weights)

            batch_loss = np.sum(loss)/param.batch_size

            training_loss += batch_loss

            # print(np.mean(o11), np.mean(o12), np.mean(o13))
            # print(np.mean(o21), np.mean(o22), np.mean(o23))
            # print(o32[0])
            # print()

            writer.add_summary(summary, epoch * total_batch + i)

        training_loss = training_loss / total_batch

        # validating by sampling every epoch


        val_loss, o11, o12, o13, o21, o22, o23 = sess.run([kbc_loss, kbc_loss1, h_r_t_pos, h_r_t_neg, kbc_loss2, t_r_h_pos, t_r_h_neg],
                            feed_dict={r_input: rel_valid['structure'],
                                       r_image_input: rel_valid['image'],
                                       r_image_padding: rel_valid['padding_image'],
                                       r_video_input: rel_valid['video'],
                                       r_video_padding: rel_valid['padding_video'],
                                       r_audio_input: rel_valid['audio'],
                                       r_audio_padding: rel_valid['padding_audio'],

                                       h_pos_structure_input: head_valid['structure'],
                                       h_pos_text_input: head_valid['text'],
                                       h_pos_text_padding: head_valid['padding_text'],
                                       h_pos_image_input: head_valid['image'],
                                       h_pos_image_padding: head_valid['padding_image'],
                                       h_pos_video_input: head_valid['video'],
                                       h_pos_video_padding: head_valid['padding_video'],
                                       h_pos_audio_input: head_valid['audio'],
                                       h_pos_audio_padding: head_valid['padding_audio'],

                                       t_pos_structure_input: tail_valid['structure'],
                                       t_pos_text_input: tail_valid['text'],
                                       t_pos_text_padding: tail_valid['padding_text'],
                                       t_pos_image_input: tail_valid['image'],
                                       t_pos_image_padding: tail_valid['padding_image'],
                                       t_pos_video_input: tail_valid['video'],
                                       t_pos_video_padding: tail_valid['padding_video'],
                                       t_pos_audio_input: tail_valid['audio'],
                                       t_pos_audio_padding: tail_valid['padding_audio'],

                                       t_neg_structure_input: tail_neg_valid['structure'],
                                       t_neg_text_input: tail_neg_valid['text'],
                                       t_neg_text_padding: tail_neg_valid['padding_text'],
                                       t_neg_image_input: tail_neg_valid['image'],
                                       t_neg_image_padding: tail_neg_valid['padding_image'],
                                       t_neg_video_input: tail_neg_valid['video'],
                                       t_neg_video_padding: tail_neg_valid['padding_video'],
                                       t_neg_audio_input: tail_neg_valid['audio'],
                                       t_neg_audio_padding: tail_neg_valid['padding_audio'],

                                       h_neg_structure_input: head_neg_valid['structure'],
                                       h_neg_text_input: head_neg_valid['text'],
                                       h_neg_text_padding: head_neg_valid['padding_text'],
                                       h_neg_image_input: head_neg_valid['image'],
                                       h_neg_image_padding: head_neg_valid['padding_image'],
                                       h_neg_video_input: head_neg_valid['video'],
                                       h_neg_video_padding: head_neg_valid['padding_video'],
                                       h_neg_audio_input: head_neg_valid['audio'],
                                       h_neg_audio_padding: head_neg_valid['padding_audio'],

                                       keep_prob: 1
                                       })

        # print(np.mean(o11), np.mean(o12), np.mean(o13))
        # print(np.mean(o21), np.mean(o22), np.mean(o23))

        val_score = np.sum(val_loss) / len(valid_data)


        print("Epoch:", (epoch + 1), "loss=", str(round(training_loss, 4)), "val_loss", str(round(val_score, 4)))

        if val_score < initial_valid_loss :
            saver.save(sess, param.model_weights_best_valid_file)
            log_file.write("save model best validation loss: " + str(initial_valid_loss) + "==>" + str(val_score) + "\n")
            print("save model valid loss: ", str(initial_valid_loss), "==>", str(val_score))
            initial_valid_loss = val_score

        loss_log.append({
            'epoch': (epoch + 1),
            'loss': round(training_loss, 4),
            'val_loss': round(val_score, 4),
            'best_val_loss': initial_valid_loss,
        })


        saver.save(sess, param.model_current_weights_file)

        log_file.write("Epoch:\t" + str(epoch + 1) + "\tloss:\t" + str(round(training_loss, 5)) + "\tval_loss:\t" + str(
            round(val_score, 5)) + "\n")
        log_file.flush()

with open(param.loss_file, 'w') as f:
    f.write(json.dumps(loss_log))

