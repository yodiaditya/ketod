# Copyright (c) Meta Platforms, Inc. and its affiliates.
#!/bin/bash

# you can use this to specify which gpu to use like 0 or 0,1 or 2,3
export CUDA_VISIBLE_DEVICES=0

# Change this with your own KETOD project path
root_path="/home/ubuntu/chat/ketod/"

# train
python3 train_simpletod.py \
--model_save_name=model1_rand \
--output_path="${root_path}outputs/" \
--input="${root_path}outputs/model1/model1.lm.rand.input.train_final.txt" \
--dev_input="${root_path}outputs/model1/model1.lm.rand.input.dev_final.txt" \
--eos_token_id=50256 \
--batch_size=12 \
--max_epoch=50 \
--learning_rate=1e-4 \
--report_loss=100 \
--report=500 \
--max_seq_len=512 \
--neg_sample \
--neg_sample_rate=3

# test gold
# python3 test_simpletod_simple.py \
# --saved_model_path="model1_rand_20231027120607/saved_model/loads/21/model.pt" \
# --output_path="${root_path}outputs/" \
# --model_dir_name="model1_gold_kg_rand" \
# --test_input="${root_path}outputs/model1/model1.lm.input.eval.test_final.txt" \
# --test_input_gold_action="${root_path}outputs/model1/model1.lm.rand.input.eval.goldaction.test_final.txt" \
# --test_input_gold_kg="${root_path}outputs/model1/model1.lm.rand.input.eval.goldkg.test_final.txt" \
# --test_input_gold_decision="${root_path}outputs/model1/model1.lm.rand.input.eval.golddecision.test_final.txt" \
# --test_oracle_input="${root_path}outputs/model1/model1.lm.rand.input.test_final.txt" \
# --test_input_original="${root_path}outputs/model1/processed_model1_test_final.json" \
# --test_inter="${root_path}outputs/model1/test_gold_inter.json" \
# --test_inter_res="${root_path}outputs/model1/test_gold_inter.json" \
# --en_schema="${root_path}outputs/model1/schema.json" \
# --num_passages=2 \
# --num_para=2 \
# --eos_token_id=50256 \
# --batch_size=1 \
# --max_seq_len=1024 \
# --gold_kg 


# # test gold
# python3 test_simpletod_simple.py \
# --saved_model_path="model1_20210818161419/saved_model/loads/11/model.pt" \
# --output_path="${root_path}outputs/" \
# --model_dir_name="model1_gold_kg_1024" \
# --test_input="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.test_gold.txt" \
# --test_input_gold_action="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.goldaction.test_gold.txt" \
# --test_input_gold_kg="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.goldkg.test_gold.txt" \
# --test_input_gold_decision="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.golddecision.test_gold.txt" \
# --test_oracle_input="${root_path}todkg_dataset/runs/model1/model1.lm.input.test_gold.txt" \
# --test_input_original="${root_path}todkg_dataset/runs/model1/processed_model1_test_gold.json" \
# --test_inter="${root_path}todkg_dataset/runs/model1/test_gold_inter.json" \
# --test_inter_res="${root_path}todkg_dataset/runs/model1/predictions_kg_select.json" \
# --en_schema="${root_path}todkg_dataset/entity_schemas/schema_all.json" \
# --num_passages=2 \
# --num_para=2 \
# --eos_token_id=50256 \
# --batch_size=1 \
# --max_seq_len=1024 \
# --gold_kg \


# # test retrieved
# python3 test_simpletod_simple.py \
# --saved_model_path="model1_20210818161419/saved_model/loads/11/model.pt" \
# --output_path="${root_path}outputs/" \
# --model_dir_name="model1_gold_action_retrieved_kg_gold_decision" \
# --test_input="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.test_retrieved.txt" \
# --test_input_gold_action="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.goldaction.test_retrieved.txt" \
# --test_input_gold_kg="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.goldkg.test_retrieved.txt" \
# --test_input_gold_decision="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.golddecision.test_retrieved.txt" \
# --test_oracle_input="${root_path}todkg_dataset/runs/model1/model1.lm.input.test_retrieved.txt" \
# --test_input_original="${root_path}todkg_dataset/runs/model1/processed_model1_test_retrieved.json" \
# --test_inter="${root_path}todkg_dataset/runs/model1/test_retrieved_inter.json" \
# --test_inter_res="${root_path}todkg_dataset/runs/model1/predictions_kg_select.json" \
# --en_schema="${root_path}todkg_dataset/entity_schemas/schema_all.json" \
# --num_passages=2 \
# --num_para=2 \
# --eos_token_id=50256 \
# --batch_size=1 \
# --max_seq_len=1024 \
# --gold_action \


# # test retrieved tfidf
# python3 test_simpletod_simple.py \
# --saved_model_path="model1_20210818161419/saved_model/loads/11/model.pt" \
# --output_path="${root_path}outputs/" \
# --model_dir_name="model1_gold_action_tfidf_kg_gold_decision" \
# --test_input="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.test_final_tfidf.txt" \
# --test_input_gold_action="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.goldaction.test_final_tfidf.txt" \
# --test_input_gold_kg="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.goldkg.test_final_tfidf.txt" \
# --test_input_gold_decision="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.golddecision.test_final_tfidf.txt" \
# --test_oracle_input="${root_path}todkg_dataset/runs/model1/model1.lm.input.test_final_tfidf.txt" \
# --test_input_original="${root_path}todkg_dataset/runs/model1/processed_model1_test_final_tfidf.json" \
# --test_inter="${root_path}todkg_dataset/runs/model1/test_final_tfidf_inter.json" \
# --test_inter_res="${root_path}todkg_dataset/runs/model1/predictions_kg_select.json" \
# --en_schema="${root_path}todkg_dataset/entity_schemas/schema_all.json" \
# --num_passages=2 \
# --num_para=2 \
# --eos_token_id=50256 \
# --batch_size=1 \
# --max_seq_len=1024 \
# --gold_action \


# # test all
# python3 test_simpletod_simple.py \
# --saved_model_path="model1_20210818161419/saved_model/loads/11/model.pt" \
# --output_path="${root_path}outputs/" \
# --model_dir_name="model1_all" \
# --test_input="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.test_gold.txt" \
# --test_input_gold_action="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.goldaction.test_gold.txt" \
# --test_input_gold_kg="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.goldkg.test_gold.txt" \
# --test_input_gold_decision="${root_path}todkg_dataset/runs/model1/model1.lm.input.eval.golddecision.test_gold.txt" \
# --test_oracle_input="${root_path}todkg_dataset/runs/model1/model1.lm.input.test_gold.txt" \
# --test_input_original="${root_path}todkg_dataset/runs/model1/processed_model1_test_gold.json" \
# --test_inter="${root_path}todkg_dataset/runs/model1/test_gold_inter.json" \
# --test_inter_res="${root_path}outputs/inference_only_20210819215118_kg_select_bert_base_model1_all/results/test/predictions.json" \
# --en_schema="${root_path}todkg_dataset/entity_schemas/schema_all.json" \
# --num_passages=2 \
# --num_para=2 \
# --eos_token_id=50256 \
# --batch_size=1 \
# --max_seq_len=1024 \
# # --gold_kg \
