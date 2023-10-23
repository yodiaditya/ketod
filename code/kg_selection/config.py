# Copyright (c) Meta Platforms, Inc. and its affiliates.
import os

class parameters():
    # prog_name = "retriever"
    root = os.path.abspath("../..")

    # using original_data_unitable
    root_path = root + "/ketod_release/"
    output_path = root + "/outputs/"

    # cache for bert model
    cache_dir = root + "/tmp/"

    model_save_name = "kg_select_bert_base_"

    train_file = root_path + "processed_kg_select_train_final.json"
    valid_file = root_path + "processed_kg_select_dev_final.json"
    test_file = root_path + "processed_kg_select_test_final.json"

    # simpletod_path = "/data/users/zhiyuchen/todkg_dataset/runs/model2_new/"
    # # test_file = root_path + "dataset/test.json"
    # # test_file = root_path + "dataset/train.json"
    # test_file = simpletod_path + "test_all_inter.json"
    # test_file = root_path + "processed_kg_select_dev_final.json"

    # model choice: bert, roberta, albert
    pretrained_model = "bert"
    model_size = "bert-base-cased"

    # pretrained_model = "roberta"
    # model_size = "roberta-large"

    # train or test
    device = "cuda"
    mode = "train"
    resume_model_path = ""
    saved_model_path = output_path + "kg_select_bert_base__20231023181001/saved_model/loads/1/model.pt"
    build_summary = False

    option = "rand"
    neg_rate = 3
    topn = 3

    # threshold for select snippets
    thresh = 0.5

    # check  on utils.py line 167 to setting tillaction_gold and others
    # entity_passages_sents_pred is for simpletod, not for KG selection
    # this is for valid and test, and we doesn't have `entity_passages_sents_pred` data
    tillaction_gold = True
    generate_all = True
    if_fill_train = True

    generate_all_neg_max = 30

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512
    dropout_rate = 0.1

    batch_size = 48
    batch_size_test = 16
    epoch = 10
    learning_rate = 3e-5

    report = 1000
    report_loss = 200
