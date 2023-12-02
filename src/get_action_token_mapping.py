from transformers import (
    AutoProcessor,
)

# model = 'Salesforce/blip2-opt-2.7b'
model = 'Salesforce/blip2-flan-t5-xl'
model_name = model.split('/')[-1]

processor = AutoProcessor.from_pretrained(
            model,
            use_fast=False,
        )

processor.tokenizer('123456 800')
vocb = processor.tokenizer.get_vocab()

# count tokens with numbers also include e.g. ▁4
count = 0
bin_nr = 0
dict_number_tokens = {'org_tokens': {}, 'action_bin_tokens': {}, 'tokens_action_bin': {}}
for key, token in vocb.items():
    if key.isdigit() and len(key) > 1:
        if count == 0:
            dict_number_tokens['org_tokens'][key] = token
            dict_number_tokens['action_bin_tokens']["<START_ACTION>"] = token
            dict_number_tokens['tokens_action_bin'][token] = "<START_ACTION>"
        elif count == 1:
            dict_number_tokens['org_tokens'][key] = token
            dict_number_tokens['action_bin_tokens']["<END_ACTION>"] = token
            dict_number_tokens['tokens_action_bin'][token] = "<END_ACTION>"
        else:
            dict_number_tokens['org_tokens'][key] = token
            dict_number_tokens['action_bin_tokens'][bin_nr] = token
            dict_number_tokens['tokens_action_bin'][token] = bin_nr
            bin_nr += 1

        count += 1
        
        
    # if key.startswith('▁') and key[1:].isdigit():
    #     dict_number_tokens['org_tokens'][key] = token
    #     dict_number_tokens['action_bin_tokens'][count] = token
    #     count += 1

#write vocab to file
with open('vocab.txt', 'w') as f:
    for key in vocb.keys():
        f.write(key + '\n')

# save dict with number tokens as json
import json
with open(f'action_bin_token_mapping_{model_name}.json', 'w') as f:
    json.dump(dict_number_tokens, f, indent=4)

import pickle
with open(f'action_bin_token_mapping_{model_name}.pkl', 'wb') as f:
    pickle.dump(dict_number_tokens, f)

print('2')