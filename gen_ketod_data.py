'''

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
'''
import argparse
import collections
import json
import os
import re
import sys
import multiprocessing

# generate ketod dataset
# python 3.9

def gen_ketod(json_in, sgd_folder_in, json_out, mode="train"):
    '''
    generate the ketod dataset
    combining our annotation and the sgd data
    '''

    # load ketod data
    with open(json_in) as f_in:
        data = json.load(f_in)

    # we will use this to match the dialogue_id for dev and test data over train data
    all_sgd_train = {}
    sgd_folder = os.path.join(sgd_folder_in, "train")

    for filename in os.listdir(sgd_folder):
        # iterate over files with name containing "dialogues" ~ dialogues_001.json, etc/
        if "dialogues" in filename:
            with open(os.path.join(sgd_folder, filename)) as f_in:
                # load the json
                this_data = json.load(f_in)

                ## iterate over each dialogue. Example of JSON:
                # dialogue_id": "1_00001",
                #    "services": [
                #      "Restaurants_1"
                #    ],
                #    "turns": [
                #      {
                #        "frames": [
                ##
                for each_data in this_data:
                    # check if no duplicate dialogue_id in the sgd train data
                    assert each_data["dialogue_id"] not in all_sgd_train
                    all_sgd_train[each_data["dialogue_id"]] = each_data

    # iterate by mode arguments
    all_sgd = {}
    sgd_folder = os.path.join(sgd_folder_in, mode)
    print("Loaded mode: ", mode)

    for filename in os.listdir(sgd_folder):
        if "dialogues" in filename:
            with open(os.path.join(sgd_folder, filename)) as f_in:
                this_data = json.load(f_in)
                for each_data in this_data:
                    assert each_data["dialogue_id"] not in all_sgd
                    all_sgd[each_data["dialogue_id"]] = each_data

    # iterate over KETOD dataset
    for each_data in data:
        # try to match dialgoue_id in `each_data` KETOD with `all_sgd` which based on mode
        # mostly will be 100% found for train data, but not for dev and test
        if each_data["dialogue_id"] in all_sgd:
            this_sgd = all_sgd[each_data["dialogue_id"]]
        else:
            # if not found, then try to find from train data
            this_sgd = all_sgd_train[each_data["dialogue_id"]]

        this_final_turns = []

        # example of each_data["turns"]
        # [{'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': True, 'entity_query': [['travel : attraction name : Apsley House']], 'enriched_utter': "I found a Museum called Apsley House that you should check out. It's the London townhouse of the Dukes of Wellington and it's open to the public as a museum and art gallery.", 'kg_snippets': [20, 24], 'kg_snippets_text': ['Apsley House Apsley House, also known as Number One, London, is the London townhouse of the Dukes of Wellington.', 'Apsley House The house is now run by English Heritage and is open to the public as a museum and art gallery, exhibiting 83 paintings from the Spanish royal collection.']}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}, {'enrich': False}]

        for sgd_turn, ketod_turn in zip(this_sgd["turns"], each_data["turns"]):
            # Combine two of SGD and KETOD. Example
            # SGD Turn :  {'frames': [{'actions': [{'act': 'GOODBYE', 'canonical_values': [], 'slot': '', 'values': []}], 'service': 'Hotels_1', 'slots': []}], 'speaker': 'SYSTEM', 'utterance': 'Okay, have a nice day!'}
            # KETOD Turn :  {'enrich': False}
            # FINAL Turn :  {'frames': [{'actions': [{'act': 'GOODBYE', 'canonical_values': [], 'slot': '', 'values': []}], 'service': 'Hotels_1', 'slots': []}], 'speaker': 'SYSTEM', 'utterance': 'Okay, have a nice day!', 'enrich': False}
            final_turn = sgd_turn | ketod_turn
            this_final_turns.append(final_turn)

        # replace KETOD turns with the combined turns from DSTD and KETOD enriched
        each_data["turns"] = this_final_turns

    with open(json_out, "w") as f:
        json.dump(data, f, indent=4)

def worker(release, sgd_dataset, final_dataset, mode):
    gen_ketod(release, sgd_dataset, final_dataset, mode=mode)

if __name__ == '__main__':
    root = os.getcwd() + "/"
    ketod_release = root + "ketod_release/"

    ketod_release_train = ketod_release + "train_ketod.json"
    ketod_release_dev = ketod_release + "dev_ketod.json"
    ketod_release_test = ketod_release + "test_ketod.json"

    sgd = root + "dstc8-schema-guided-dialogue/"

    train_final = ketod_release + "train_final.json"
    dev_final = ketod_release + "dev_final.json"
    test_final = ketod_release + "test_final.json"

    # setting up processes
    processes = [
        multiprocessing.Process(target=worker, args=(ketod_release_train, sgd, train_final, "train")),
        multiprocessing.Process(target=worker, args=(ketod_release_dev, sgd, dev_final, "dev")),
        multiprocessing.Process(target=worker, args=(ketod_release_test, sgd, test_final, "test"))
    ]

    for process in processes:
        process.start()

    # ensure all process have finished execution
    for process in processes:
        process.join()
    
    print("All processes are completed.")

    # generate ketod dataset
    # gen_ketod(ketod_release_train, sgd, train_final, mode="train")
    # gen_ketod(ketod_release_dev, sgd, dev_final, mode="dev")
    # gen_ketod(ketod_release_test, sgd, test_final, mode="test")
