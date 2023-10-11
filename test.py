import operator
import os

import numpy as np
import tensorflow as tf

import test_parameters as param
import util as u

graph = tf.get_default_graph()

batch_size = 500

def predict_best_tail(test_triple, full_triple_list, full_entity_list, embeddings, all_entities, all_relations):
    results = {}
    gt_head = test_triple[0]
    gt_head_embeddings = u.load_multimodal_for_single_key(gt_head, all_entities, is_entity=True)

    gt_rel = test_triple[1]
    gt_rel_embeddings = u.load_multimodal_for_single_key(gt_rel, all_relations, is_entity=False)

    gt_tail_org = test_triple[2]
    if param.strict_relation:
        gt_tail = [gt_tail_org]
    else:
        gt_tail = u.get_correct_tails(gt_head, gt_rel, full_triple_list, embeddings['relation'])


    total_batches = len(full_entity_list)//batch_size
    if len(full_entity_list) % batch_size != 0:
        total_batches += 1

    predictions = []
    for batch_i in range(total_batches):
        start = batch_size * (batch_i)
        end = batch_size * (batch_i + 1)


        tails_embeddings_list = None

        head_embeddings_list = {}
        for k, v in gt_head_embeddings.items():
            head_embeddings_list[k] = np.tile(v, (len(full_entity_list[start:end]),1))
        full_relation_embeddings = {}
        for k, v in gt_rel_embeddings.items():
            full_relation_embeddings[k] = np.tile(v,(len(full_entity_list[start:end]),1))


        for i in range(len(full_entity_list[start:end])):

            next_data = u.load_multimodal_for_single_key(full_entity_list[start+i], all_entities, is_entity=True)
            if tails_embeddings_list == None:
                tails_embeddings_list = {}
                for k, v in next_data.items():
                   tails_embeddings_list[k] = [v]
            else:
                for k, v in tails_embeddings_list.items():
                    v.append(next_data[k])

        for dic in [head_embeddings_list, full_relation_embeddings, tails_embeddings_list]:
            for v in dic.values():
                v = np.stack(v)

        sub_predictions = predict_tail(head_embeddings_list, full_relation_embeddings, tails_embeddings_list)
        for p in sub_predictions:
            predictions.append(p)

    predictions = [predictions]
    for i in range(0, len(predictions[0])):
        if  full_entity_list[i] == gt_head  and gt_head not in gt_tail:
            pass
            #results[full_entity_list[i]] = 0
        else:
            results[full_entity_list[i]] = predictions[0][i]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    top_10_predictions = [x[0] for x in sorted_x[:10]]
    sorted_keys = [x[0] for x in sorted_x]
    index_correct_tail_raw = sorted_keys.index(gt_tail_org)

    gt_tail_to_filter = [x for x in gt_tail if x != gt_tail_org]
    # remove the correct tails from the predictions
    for key in gt_tail_to_filter:
        del results[key]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    sorted_keys = [x[0] for x in sorted_x]
    index_tail_head_filter = sorted_keys.index(gt_tail_org)

    return (index_correct_tail_raw + 1), (index_tail_head_filter + 1), top_10_predictions

def predict_tail(head, rel, tail):


    r_input = graph.get_tensor_by_name("input/r_input:0")
    r_image_input = graph.get_tensor_by_name("input/r_image_input:0")
    r_image_padding = graph.get_tensor_by_name("input/r_image_padding:0")
    r_video_input = graph.get_tensor_by_name("input/r_video_input:0")
    r_video_padding = graph.get_tensor_by_name("input/r_video_padding:0")
    r_audio_input = graph.get_tensor_by_name("input/r_audio_input:0")
    r_audio_padding = graph.get_tensor_by_name("input/r_audio_padding:0")

    h_pos_structure_input = graph.get_tensor_by_name("input/h_pos_structure_input:0")
    h_pos_text_input = graph.get_tensor_by_name("input/h_pos_text_input:0")
    h_pos_text_padding = graph.get_tensor_by_name("input/h_pos_text_padding:0")
    h_pos_image_input = graph.get_tensor_by_name("input/h_pos_image_input:0")
    h_pos_image_padding = graph.get_tensor_by_name("input/h_pos_image_padding:0")
    h_pos_video_input = graph.get_tensor_by_name("input/h_pos_video_input:0")
    h_pos_video_padding = graph.get_tensor_by_name("input/h_pos_video_padding:0")
    h_pos_audio_input = graph.get_tensor_by_name("input/h_pos_audio_input:0")
    h_pos_audio_padding = graph.get_tensor_by_name("input/h_pos_audio_padding:0")

    t_pos_structure_input = graph.get_tensor_by_name("input/t_pos_structure_input:0")
    t_pos_text_input = graph.get_tensor_by_name("input/t_pos_text_input:0")
    t_pos_text_padding = graph.get_tensor_by_name("input/t_pos_text_padding:0")
    t_pos_image_input = graph.get_tensor_by_name("input/t_pos_image_input:0")
    t_pos_image_padding = graph.get_tensor_by_name("input/t_pos_image_padding:0")
    t_pos_video_input = graph.get_tensor_by_name("input/t_pos_video_input:0")
    t_pos_video_padding = graph.get_tensor_by_name("input/t_pos_video_padding:0")
    t_pos_audio_input = graph.get_tensor_by_name("input/t_pos_audio_input:0")
    t_pos_audio_padding = graph.get_tensor_by_name("input/t_pos_audio_padding:0")


    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

    h_r_t_pos = graph.get_tensor_by_name("cosine/h_r_t_pos:0")


    predictions = h_r_t_pos.eval(feed_dict={r_input: rel['structure'],
                                            r_image_input: rel['image'],
                                            r_image_padding: rel['padding_image'],
                                            r_video_input: rel['video'],
                                            r_video_padding: rel['padding_video'],
                                            r_audio_input: rel['audio'],
                                            r_audio_padding: rel['padding_audio'],

                                            h_pos_structure_input: head['structure'],
                                            h_pos_text_input: head['text'],
                                            h_pos_text_padding: head['padding_text'],
                                            h_pos_image_input: head['image'],
                                            h_pos_image_padding: head['padding_image'],
                                            h_pos_video_input: head['video'],
                                            h_pos_video_padding: head['padding_video'],
                                            h_pos_audio_input: head['audio'],
                                            h_pos_audio_padding: head['padding_audio'],

                                            t_pos_structure_input: tail['structure'],
                                            t_pos_text_input: tail['text'],
                                            t_pos_text_padding: tail['padding_text'],
                                            t_pos_image_input: tail['image'],
                                            t_pos_image_padding: tail['padding_image'],
                                            t_pos_video_input: tail['video'],
                                            t_pos_video_padding: tail['padding_video'],
                                            t_pos_audio_input: tail['audio'],
                                            t_pos_audio_padding: tail['padding_audio'],
                                            keep_prob: 1.0})
    return predictions



def predict_head(head, rel, tail):


    r_input = graph.get_tensor_by_name("input/r_input:0")
    r_image_input = graph.get_tensor_by_name("input/r_image_input:0")
    r_image_padding = graph.get_tensor_by_name("input/r_image_padding:0")
    r_video_input = graph.get_tensor_by_name("input/r_video_input:0")
    r_video_padding = graph.get_tensor_by_name("input/r_video_padding:0")
    r_audio_input = graph.get_tensor_by_name("input/r_audio_input:0")
    r_audio_padding = graph.get_tensor_by_name("input/r_audio_padding:0")

    h_pos_structure_input = graph.get_tensor_by_name("input/h_pos_structure_input:0")
    h_pos_text_input = graph.get_tensor_by_name("input/h_pos_text_input:0")
    h_pos_text_padding = graph.get_tensor_by_name("input/h_pos_text_padding:0")
    h_pos_image_input = graph.get_tensor_by_name("input/h_pos_image_input:0")
    h_pos_image_padding = graph.get_tensor_by_name("input/h_pos_image_padding:0")
    h_pos_video_input = graph.get_tensor_by_name("input/h_pos_video_input:0")
    h_pos_video_padding = graph.get_tensor_by_name("input/h_pos_video_padding:0")
    h_pos_audio_input = graph.get_tensor_by_name("input/h_pos_audio_input:0")
    h_pos_audio_padding = graph.get_tensor_by_name("input/h_pos_audio_padding:0")

    t_pos_structure_input = graph.get_tensor_by_name("input/t_pos_structure_input:0")
    t_pos_text_input = graph.get_tensor_by_name("input/t_pos_text_input:0")
    t_pos_text_padding = graph.get_tensor_by_name("input/t_pos_text_padding:0")
    t_pos_image_input = graph.get_tensor_by_name("input/t_pos_image_input:0")
    t_pos_image_padding = graph.get_tensor_by_name("input/t_pos_image_padding:0")
    t_pos_video_input = graph.get_tensor_by_name("input/t_pos_video_input:0")
    t_pos_video_padding = graph.get_tensor_by_name("input/t_pos_video_padding:0")
    t_pos_audio_input = graph.get_tensor_by_name("input/t_pos_audio_input:0")
    t_pos_audio_padding = graph.get_tensor_by_name("input/t_pos_audio_padding:0")


    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

    t_r_h_pos = graph.get_tensor_by_name("cosine/t_r_h_pos:0")

    predictions = t_r_h_pos.eval(feed_dict={r_input: rel['structure'],
                                            r_image_input: rel['image'],
                                            r_image_padding: rel['padding_image'],
                                            r_video_input: rel['video'],
                                            r_video_padding: rel['padding_video'],
                                            r_audio_input: rel['audio'],
                                            r_audio_padding: rel['padding_audio'],

                                            h_pos_structure_input: head['structure'],
                                            h_pos_text_input: head['text'],
                                            h_pos_text_padding: head['padding_text'],
                                            h_pos_image_input: head['image'],
                                            h_pos_image_padding: head['padding_image'],
                                            h_pos_video_input: head['video'],
                                            h_pos_video_padding: head['padding_video'],
                                            h_pos_audio_input: head['audio'],
                                            h_pos_audio_padding: head['padding_audio'],

                                            t_pos_structure_input: tail['structure'],
                                            t_pos_text_input: tail['text'],
                                            t_pos_text_padding: tail['padding_text'],
                                            t_pos_image_input: tail['image'],
                                            t_pos_image_padding: tail['padding_image'],
                                            t_pos_video_input: tail['video'],
                                            t_pos_video_padding: tail['padding_video'],
                                            t_pos_audio_input: tail['audio'],
                                            t_pos_audio_padding: tail['padding_audio'],
                                            keep_prob: 1.0})




    return predictions



def predict_best_head(test_triple, full_triple_list, full_entity_list, embeddings, all_entities, all_relations):

    #triple: head, tail, relation
    results = {}
    gt_tail = test_triple[2] #tail
    gt_tail_embeddings = u.load_multimodal_for_single_key(gt_tail, all_entities, is_entity=True)

    gt_rel = test_triple[1]
    gt_rel_embeddings = u.load_multimodal_for_single_key(gt_rel, all_relations, is_entity=False)

    gt_head_org = test_triple[0]
    if param.strict_relation:
        gt_head = [gt_tail_org]
    else:
        gt_head = u.get_correct_heads(gt_tail, gt_rel, full_triple_list, embeddings['relation'])



    total_batches = len(full_entity_list)//batch_size
    if len(full_entity_list) % batch_size != 0:
        total_batches += 1

    predictions = []
    for batch_i in range(total_batches):
        start = batch_size * (batch_i)
        end = batch_size * (batch_i + 1)

        heads_embeddings_list = None

        tail_embeddings_list = {}
        for k, v in gt_tail_embeddings.items():
            tail_embeddings_list[k] = np.tile(v, (len(full_entity_list[start:end]),1))
        full_relation_embeddings = {}
        for k, v in gt_rel_embeddings.items():
            full_relation_embeddings[k] = np.tile(v,(len(full_entity_list[start:end]),1))


        for i in range(len(full_entity_list[start:end])):
            next_data = u.load_multimodal_for_single_key(full_entity_list[start+i], all_entities, is_entity=True)
            if heads_embeddings_list == None:
                heads_embeddings_list = {}
                for k, v in next_data.items():
                   heads_embeddings_list[k] = [v]
            else:
                for k, v in heads_embeddings_list.items():
                    v.append(next_data[k])

        for dic in [heads_embeddings_list, full_relation_embeddings, tail_embeddings_list]:
            for v in dic.values():
                v = np.stack(v)

        sub_predictions = predict_head(heads_embeddings_list, full_relation_embeddings, tail_embeddings_list)

        for p in sub_predictions:
            predictions.append(p)


    predictions = [predictions]

    for i in range(0, len(predictions[0])):
        if full_entity_list[i] == gt_tail  and gt_tail not in gt_head:

        #    #results[full_entity_list[i]] = 0
            pass
        else:
            results[full_entity_list[i]] = predictions[0][i]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    top_10_predictions = [x[0] for x in sorted_x[:10]]
    sorted_keys = [x[0] for x in sorted_x]
    index_correct_head_raw = sorted_keys.index(gt_head_org)

    gt_tail_to_filter = [x for x in gt_head if x != gt_head_org]
    # remove the correct tails from the predictions
    for key in gt_tail_to_filter:
        del results[key]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    sorted_keys = [x[0] for x in sorted_x]
    index_head_filter = sorted_keys.index(gt_head_org)

    return (index_correct_head_raw + 1), (index_head_filter + 1), top_10_predictions

############ Testing Part #######################
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

all_entities = {}
all_relations = {}

_ = u.load_full_data(param.all_triples_file, embeddings, all_entities, all_relations)

all_triples, entity_list = u.load_training_triples(param.all_triples_file)

print("#Entities", len(entity_list))
all_test_triples, _ = u.load_training_triples(param.test_triples_file)
#all_test_triples = all_test_triples[:1000]
print("#Test triples", len(all_test_triples))  # Triple: head, tail, relation


tail_mrr_raw = 0
tail_mrr_filter = 0
tail_hits1_raw = 0
tail_hits1_filter = 0
tail_hits3_raw = 0
tail_hits3_filter = 0
tail_hits10_raw = 0
tail_hits10_filter = 0
head_mrr_raw = 0
head_mrr_filter = 0
head_hits1_raw = 0
head_hits1_filter = 0
head_hits3_raw = 0
head_hits3_filter = 0
head_hits10_raw = 0
head_hits10_filter = 0

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
        #print("Model restored from file: %s" % param.current_model_meta_file)
        mrr_raw = 0.0
        mrr_filter = 0.0
        hits_at_1_raw = 0.0
        hits_at_1_filter = 0.0
        hits_at_3_raw = 0.0
        hits_at_3_filter = 0.0
        hits_at_10_raw = 0.0
        hits_at_10_filter = 0.0
        lines = []

        #new_saver = tf.train.import_meta_graph(param.model_meta_file)
        # new_saver.restore(sess, param.model_weights_best_file)

        saver = tf.train.import_meta_graph(param.best_valid_model_meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(param.checkpoint_best_valid_dir))

        graph = tf.get_default_graph()
        #Warning only for relation classification
        #entity_list = u.load_relation_list(param.all_triples_file, entity_embeddings)
        counter = 1
        for triple in all_test_triples:
            rank_raw, rank_filter, top_10 = predict_best_tail(triple, all_triples, entity_list, embeddings, all_entities, all_relations)

            line = triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\t" + str(top_10) + "\t" + str(rank_raw) + "\t" + str(
                    rank_filter) + "\n"

            #print(line)
            lines.append(line)
            print(str(counter) + "/" + str(len(all_test_triples)) + " " + str(rank_raw) + " " + str(rank_filter))
            counter +=1
            mrr_raw += 1.0 / rank_raw
            mrr_filter += 1.0 / rank_filter
            if rank_raw <= 1:
                hits_at_1_raw += 1
            if rank_filter <= 1:
                hits_at_1_filter += 1
            if rank_raw <= 3:
                hits_at_3_raw += 1
            if rank_filter <= 3:
                hits_at_3_filter += 1
            if rank_raw <= 10:
                hits_at_10_raw += 1
            if rank_filter <= 10:
                hits_at_10_filter += 1

        mrr_raw /= len(all_test_triples) / 100
        mrr_filter /= len(all_test_triples) / 100
        hits_at_1_raw /= len(all_test_triples) / 100
        hits_at_1_filter /= len(all_test_triples) / 100
        hits_at_3_raw /= len(all_test_triples) / 100
        hits_at_3_filter /= len(all_test_triples) / 100
        hits_at_10_raw /= len(all_test_triples) / 100
        hits_at_10_filter /= len(all_test_triples) / 100

        print("MRR Raw", mrr_raw, "MRR Filter", mrr_filter)
        print("Hits@1 Raw", hits_at_1_raw, "Hits@1 Filter", hits_at_1_filter)
        print("Hits@3 Raw", hits_at_3_raw, "Hits@3 Filter", hits_at_3_filter)
        print("Hits@10 Raw", hits_at_10_raw, "Hits@10 Filter", hits_at_10_filter)

        # Write to a file
        #results_file = param.result_file
        results_file = param.result_file.replace(".txt","tail_prediction.txt")

        if os.path.isfile(results_file):
            results_file = results_file.replace(".txt", "_1.txt")

        print("write the results into", results_file)
        with open(results_file, "w") as f:
            f.write("MRR Raw" + "\t" + str(mrr_raw) + "\t" + "MRR Filter" + "\t" + str(mrr_filter) + "\n")
            f.write("Hits@1 Raw" + "\t" + str(hits_at_1_raw) + "\t" + "Hits@1 Filter" + "\t" + str(
                hits_at_1_filter) + "\n" + "\n")
            f.write("Hits@3 Raw" + "\t" + str(hits_at_3_raw) + "\t" + "Hits@3 Filter" + "\t" + str(
                hits_at_3_filter) + "\n" + "\n")
            f.write("Hits@10 Raw" + "\t" + str(hits_at_10_raw) + "\t" + "Hits@10 Filter" + "\t" + str(
                hits_at_10_filter) + "\n" + "\n")
            f.write("Head \t Relation \t Gold Tail \t Top Predicted Tails \t Raw Rank \t Filtered Rank\n")
            for l in lines:
                f.write(l.encode('utf-8'))

        tail_mrr_raw = mrr_raw
        tail_mrr_filter = mrr_filter
        tail_hits1_raw = hits_at_1_raw
        tail_hits1_filter = hits_at_1_filter
        tail_hits3_raw = hits_at_3_raw
        tail_hits3_filter = hits_at_3_filter
        tail_hits10_raw = hits_at_10_raw
        tail_hits10_filter = hits_at_10_filter


        mrr_raw = 0.0
        mrr_filter = 0.0
        hits_at_1_raw = 0.0
        hits_at_1_filter = 0.0
        hits_at_3_raw = 0.0
        hits_at_3_filter = 0.0
        hits_at_10_raw = 0.0
        hits_at_10_filter = 0.0
        lines = []

        counter = 1
        for triple in all_test_triples:
            rank_raw, rank_filter, top_10 = predict_best_head(triple, all_triples, entity_list, embeddings, all_entities, all_relations)

            line = triple[1] + "\t" + triple[1] + "\t" + triple[2] + "\t" + str(top_10) + "\t" + str(rank_raw) + "\t" + str(
                    rank_filter) + "\n"

            #print(line)
            lines.append(line)
            print(str(counter) + "/" + str(len(all_test_triples)) + " " + str(rank_raw) + " " + str(rank_filter))
            counter += 1
            mrr_raw += 1.0 / rank_raw
            mrr_filter += 1.0 / rank_filter
            if rank_raw <= 1:
                hits_at_1_raw += 1
            if rank_filter <= 1:
                hits_at_1_filter += 1
            if rank_raw <= 3:
                hits_at_3_raw += 1
            if rank_filter <= 3:
                hits_at_3_filter += 1
            if rank_raw <= 10:
                hits_at_10_raw += 1
            if rank_filter <= 10:
                hits_at_10_filter += 1

        mrr_raw /= len(all_test_triples) / 100
        mrr_filter /= len(all_test_triples) / 100
        hits_at_1_raw /= len(all_test_triples) / 100
        hits_at_1_filter /= len(all_test_triples) / 100
        hits_at_3_raw /= len(all_test_triples) / 100
        hits_at_3_filter /= len(all_test_triples) / 100
        hits_at_10_raw /= len(all_test_triples) / 100
        hits_at_10_filter /= len(all_test_triples) / 100

        print("MRR Raw", mrr_raw, "MRR Filter", mrr_filter)
        print("Hits@1 Raw", hits_at_1_raw, "Hits@1 Filter", hits_at_1_filter)
        print("Hits@3 Raw", hits_at_3_raw, "Hits@3 Filter", hits_at_3_filter)
        print("Hits@10 Raw", hits_at_10_raw, "Hits@10 Filter", hits_at_10_filter)

        # Write to a file
        results_file = param.result_file.replace(".txt","head_prediction.txt")
        if os.path.isfile(results_file):
            results_file = results_file.replace(".txt", "_1.txt")

        print("write the results into", results_file)
        with open(results_file, "w") as f:
            f.write("MRR Raw" + "\t" + str(mrr_raw) + "\t" + "MRR Filter" + "\t" + str(mrr_filter) + "\n")
            f.write("Hits@1 Raw" + "\t" + str(hits_at_1_raw) + "\t" + "Hits@1 Filter" + "\t" + str(
                hits_at_1_filter) + "\n" + "\n")
            f.write("Hits@3 Raw" + "\t" + str(hits_at_3_raw) + "\t" + "Hits@3 Filter" + "\t" + str(
                hits_at_3_filter) + "\n" + "\n")
            f.write("Hits@10 Raw" + "\t" + str(hits_at_10_raw) + "\t" + "Hits@10 Filter" + "\t" + str(
                hits_at_10_filter) + "\n" + "\n")
            f.write("Tail \t Relation \t Gold Head \t Top Predicted Heads \t Raw Rank \t Filtered Rank\n")
            for l in lines:
                f.write(l.encode('utf-8'))

        head_mrr_raw = mrr_raw
        head_mrr_filter = mrr_filter
        head_hits1_raw = hits_at_1_raw
        head_hits1_filter = hits_at_1_filter
        head_hits3_raw = hits_at_3_raw
        head_hits3_filter = hits_at_3_filter
        head_hits10_raw = hits_at_10_raw
        head_hits10_filter = hits_at_10_filter

print("+++++++++++++++ Evaluation Summary ++++++++++++++++")
print("MRR Raw Tail \t MRR Filter Tail \t Hits@1 Raw Tail \t Hits@1 Filter Tail \t Hits@3 Raw Tail \t Hits@3 Filter Tail \t Hits@10 Raw Tail \t Hits@10 Filter Tail")
print(str(tail_mrr_raw)+"\t"+str(tail_mrr_filter)+"\t"+str(tail_hits1_raw)+"\t"+str(tail_hits1_filter)+"\t"+str(tail_hits3_raw)+"\t"+str(tail_hits3_filter)+"\t"+str(tail_hits10_raw)+"\t"+str(tail_hits10_filter))


print("MRR Raw Head \t MRR Filter Head \t Hits@1 Raw Head \t Hits@1 Filter Head \t Hits@3 Raw Head \t Hits@3 Filter Head \t Hits@10 Raw Head \t Hits@10 Filter Head")
print(str(head_mrr_raw)+"\t"+str(head_mrr_filter)+"\t"+str(head_hits1_raw)+"\t"+str(head_hits1_filter)+"\t"+str(head_hits3_raw)+"\t"+str(head_hits3_filter)+"\t"+str(head_hits10_raw)+"\t"+str(head_hits10_filter))
