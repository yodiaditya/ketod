from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch import nn
from colorama import Fore, Back, Style
import sys, os
import time

if __name__ == "__main__":
    print('\33]0;KETOD\a', end='')
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:0"

    model_checkpoint = sys.argv[1]
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<PAD>')

    tokenizer.add_special_tokens({'additional_special_tokens': ['<|belief|>', '<|endofbelief|>', '<|action|>', '<|endofaction|>', \
                                                                '<|response|>', '<|endofresponse|>', '<|context|>', '<|endofcontext|>', '<|user|>', '<|system|>', \
                                                                '<|task|>', '<|endoftask|>', '<|chitchat|>', '<|nochitchat|>', '<|endofdecision|>', '<|knowledge|>', '<|endofknowledge|>', '<|dbresults|>', '<|endofdbresults|>']})

    model.resize_token_embeddings(len(tokenizer))
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_checkpoint))
    print("model loaded")

    model.to(device)
    model.eval()

    MAX_LEN = 512

    print(Fore.MAGENTA + "\nKETOD ready to chat. How I can serve you?")

    history = []
    context = ""
    input_text = ""
    turn = 0 

    while True:
        print(Fore.WHITE + "", end="")
        raw_text = input("You: ")

        input_text = raw_text.replace('You: ', '')
        
        if input_text in ["quit", "exit", "bye", "q"]:
            print(Fore.MAGENTA + "KETOD: Bye! Have a nice day!")
            break

        user = '<|user|> {}'.format(input_text)
        context = context + ' ' + user
        text = '<|context|> {} <|endofcontext|>'.format(context)

        print("Context: ", context.strip())

        # IMPORTANT: convert it into list to make it works
        text = [text.strip()]
        eos_token_id = tokenizer.encode("<|endofresponse|>")[0]
        response = []

        with torch.no_grad():
            encodings_dict = tokenizer.batch_encode_plus(text, padding=True)

            input_ids = torch.tensor(encodings_dict['input_ids'])
            attn_mask = torch.tensor(encodings_dict['attention_mask'])

            seq_len = len(input_ids[0])

            num_tokens_to_produce = 1024 - seq_len
            pad_token_id = tokenizer.pad_token_id

            eos_not_in_sents = torch.ones(input_ids.shape[0]).long()

            last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
            start_idx = (last_non_masked_idx).view(-1, 1).repeat(1, tokenizer.vocab_size + len(tokenizer.additional_special_tokens)).unsqueeze(1)

            position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.shape[0])])
            for i, position_ids_slice in enumerate(position_ids):
                position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]

            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            eos_not_in_sents = eos_not_in_sents.to(device)
            start_idx = start_idx.to(device)
            position_ids = position_ids.to(device)

            for step in range(num_tokens_to_produce):
                outputs = model(input_ids, 
                                attention_mask=attn_mask, 
                                position_ids=position_ids)

                if step == 0:
                    next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
                else:
                    next_token_logits = outputs.logits[:, -1, :]

                next_tokens = torch.argmax(next_token_logits, dim=-1)
                eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long())
                tokens_to_add = next_tokens * (eos_not_in_sents) + pad_token_id * (1 - eos_not_in_sents)

                input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1)).long().to(device)], dim=1)
                position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

                if torch.max(eos_not_in_sents) == 0:
                    break

            response = [tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=True).replace("<|endofresponse|>", "") for output in input_ids]

        system_res = []
        for row in response:
            row = row.replace("<PAD>", "")
            row = row.strip() + " <|endofresponse|>"
            system_res.append(row)
        
        print(Fore.CYAN + "KETOD: ", end="")
        for a in system_res:
            print(a + " ", end="\n")
            sys.stdout.flush()
            time.sleep(0.5)

        print(Fore.YELLOW + "KETOD: ", end="")

        final_response = system_res[0].split("<|response|>")[1].split("<|endofresponse|>")[0]
        
        for a in final_response:
            print('\033[1m' + a + "", end="")
            sys.stdout.flush()
            time.sleep(0.03)
        
        print("\n")

