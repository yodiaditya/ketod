# KETOD: Knowledge-Enriched Task-Oriented Dialogue

This repo contains the dataset from NAACL 2022 paper "KETOD: Knowledge-Enriched Task-Oriented Dialogue"
<https://arxiv.org/abs/2205.05589>

Notes: I made this modification to works with latest version of pytorch and huggingface transformers.

## Requirements
Make sure to install Pytorch https://pytorch.org/get-started/locally and another dependencies packages
```
pip install -r requirements.txt
```

## Dataset Generation
If you never heard about Google SGD dataset, please read it here: https://blog.research.google/2019/10/introducing-schema-guided-dialogue.html   
KETOD is built upon the google SGD dataset and combined with author knowledge-enriched utterances annotations and the script to generate the final dataset. 

To generate the full KETOD dataset in root folder, execute this:
```
python gen_ketod_data.py 
```

Notes: this will combine between SGD dataset and `enriched knowledge` dataset from author.
Three new files generated at `ketod_release` folder: `train_final.json`, `dev_final.json`, and `test_final.json`.

Have a look on my detailed comments in the file `gen_ketod_data.py` or see the dataset format below.  

## Dataset Format

Each entry of the data is one dialogue. It has the following fields:
```
"dialogue_id": unique id of the dialogue.

"turns": the list of dialogue turns. Besides the original fields in the SGD dataset, if it is an enriched turn, then we have the following additional fields:
    {
      "enrich": True. For turns without chitchat enrichment, this field is False. 
      "entity_query": The entity query we use to do knowledge retrieval.
      "enriched_utter": The utterance enriched with chitchat. Another field 'utterance' is the original response in the SGD dataset.
      "kg_snippets": the index of the ground truth knowledge snippets
      "kg_snippets_text": the content of the ground truth knowledge snippets
    }
  
"dialog_query": all the entity queries we use to do knowledge retrieval in this dialog
"entity_passages": all the wikipedia passages retrieved in this dialog
"entity_passage_sents": all the wikipedia passages retrieved in this dialog, breaked into snippets associated with index numbers
```

## Knowledge Selection Model Dataset Generation
To run the knowledge selection model, go to "kg_selection" folder and execute `process_data.py`   
Have a look my detailed comments inside the file  

```
cd code 
python process_data.py --data train_final.json
python process_data.py --data dev_final.json
python process_data.py --data test_final.json
```

And this will produce file `processed_kg_select_train_final.json`, `processed_kg_select_dev_final.json` and `processed_kg_select_test_final.json` in `ketod_release` folder   

## Train Knowledge Selection Model
Edit file `run_model.sh` and adjust your GPU. Next, edit `config_py` and adjust the `batch_size` depending on your GPU.
For example, using 2x RTX 4090, the batch size is 48 or 56.

Train the model 
```
bash run_model.sh
```

Run Test. Modify `run_model.sh` and comment `Main.py` and uncomment `Test.py`. Then run it again.

```
bash run_model.sh
```

## Running SimpleToDPlus Model
To run the SimpleToDPlus model, go to "simpletodplus" folder: modify and run gen_kg_train.py to generate data files with the kg selection results. Then run gen_data.py to generate train/dev/test files for the model input formats. Using the run_simpletod.sh script, run train_simpletod.py for training, and test_simpletod_simple.py for testing. You need to modify and follow the steps at the end of the test_simpletod_simple.py file to generate the results for each step. 

## Citation
If you find this project useful, please cite it using the following format

```
@inproceedings{DBLP:conf/naacl/ChenLMSCW22,
  author    = {Zhiyu Chen and
               Bing Liu and
               Seungwhan Moon and
               Chinnadhurai Sankar and
               Paul A. Crook and
               William Yang Wang},
  editor    = {Marine Carpuat and
               Marie{-}Catherine de Marneffe and
               Iv{\'{a}}n Vladimir Meza Ru{\'{\i}}z},
  title     = {{KETOD:} Knowledge-Enriched Task-Oriented Dialogue},
  booktitle = {Findings of the Association for Computational Linguistics: {NAACL}
               2022, Seattle, WA, United States, July 10-15, 2022},
  pages     = {2581--2593},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://doi.org/10.18653/v1/2022.findings-naacl.197},
  doi       = {10.18653/v1/2022.findings-naacl.197},
  timestamp = {Mon, 01 Aug 2022 16:27:57 +0200},
  biburl    = {https://dblp.org/rec/conf/naacl/ChenLMSCW22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License
KETOD is released under MIT license, see [LICENSE](https://github.com/facebookresearch/ketod/blob/main/LICENSE) for details.
