# Copyright (c) Meta Platforms, Inc. and its affiliates.
import json
from tqdm import tqdm
import os
from utils import get_kg_snippets_dict

'''
merge kg selection results with original
'''
def merge_train(json_in, json_out, topn_snippets=3, is_test=True):
    '''
    merge train or dev
    '''
    with open(json_in) as f:
        data_all = json.load(f)

    for each_data in tqdm(data_all):
        all_kg_snippets_dict = get_kg_snippets_dict(each_data)

        for turn in each_data["turns"]:
            if turn["speaker"] == "SYSTEM":

                this_retrieved_kg_text = []
                if turn["enrich"]:
                    if not is_test:
                        this_retrieved_kg_text = turn["kg_snippets_text"][:]

                        retrieved_ind = 0
                        while len(this_retrieved_kg_text) < topn_snippets and retrieved_ind < len(turn["retrieved"]):
                            # make up with the retrieved ones
                            this_added_ind = turn["retrieved"][retrieved_ind]
                            if this_added_ind not in turn["kg_snippets"]:
                                added_kg_text = all_kg_snippets_dict[turn["retrieved"][retrieved_ind]]
                                this_retrieved_kg_text.append(added_kg_text)
                            retrieved_ind += 1
                    else:
                        for each_added in turn["retrieved"]:
                            this_retrieved_kg_text.append(all_kg_snippets_dict[each_added])

                else:
                    for each_added in turn["retrieved"]:
                        this_retrieved_kg_text.append(all_kg_snippets_dict[each_added])

                turn["merge_retrieved"] = this_retrieved_kg_text

    with open(json_out, "w") as f:
        json.dump(data_all, f, indent=4)

parent_folder = os.path.abspath("../..")
root = parent_folder + "/outputs/"
tgt = root + "model1/"

try:
    os.makedirs(tgt, exist_ok=False)
except:
    pass

# train
# json_in = root + "inference_only_20231024221333_kg_select_bert_base_/results/test/predictions.json"
# json_out = tgt + "train_final.json"

# merge_train(json_in, json_out, topn_snippets=3, is_test=False)

# dev
json_in = root + "inference_only_20231025071409_kg_select_bert_base_/results/test/predictions.json"
json_out = tgt + "dev_final.json"

merge_train(json_in, json_out, topn_snippets=3, is_test=True)

# test
# json_in = root + "inference_only_20231025073933_kg_select_bert_base_/results/test/predictions.json"
# json_out = tgt + "test_final.json"
# merge_train(json_in, json_out, topn_snippets=3, is_test=True)
