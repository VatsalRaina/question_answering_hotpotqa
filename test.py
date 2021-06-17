#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import datetime

from transformers import ElectraTokenizer
from transformers import AdamW, ElectraConfig

from models import ElectraQA

MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--model_path', type=str, help='Load path to which trained model will be loaded from')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")
parser.add_argument('--data_path', type=str, help='Load path to evaluation data')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def _find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
    # print("Didn't find match, return <no answer>")
    return -1,0

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    with open(args.data_path) as f:
        all_samples = json.load(f)

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    input_ids = []
    attention_masks = []
    token_type_ids = []

    count = 0
    print(len(all_samples))
    for sample in all_samples:
        count+=1
        # if count > 500:
        #     break
        if count%100==0:
            print(count)
        question = sample["question"]
        context = "".join(sample["context"][0][1]) + " " + "".join(sample["context"][1][1])
        combo = question + " [SEP] " + context
        input_encodings_dict = tokenizer(combo, truncation=True, max_length=MAXLEN, padding="max_length")
        inp_ids = input_encodings_dict['input_ids']
        inp_att_msk = input_encodings_dict['attention_mask']
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]  # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token

        input_ids.append(inp_ids)
        token_type_ids.append(tok_type_ids)
        attention_masks.append(inp_att_msk)

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)


    eval_data = TensorDataset(input_ids, token_type_ids, attention_masks)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    pred_start_logits = []
    pred_end_logits = []
    pred_ansTyp_logits = []
    pred_suppFacts_logits = []
    count = 0
    print(len(eval_dataloader))
    for b_input_ids, b_tok_typ_ids, b_att_msks in eval_dataloader:
        print(count)
        count+=1
        b_input_ids, b_tok_typ_ids, b_att_msks = b_input_ids.to(device), b_tok_typ_ids.to(device), b_att_msks.to(device)
        with torch.no_grad():
            start_logits, end_logits, ansTyp_logits, suppFacts_logits = model(input_ids=b_input_ids, attention_mask=b_att_msks, token_type_ids=b_tok_typ_ids)
        b_start_logits = start_logits.detach().cpu().numpy().tolist()
        pred_start_logits += b_start_logits
        b_end_logits = end_logits.detach().cpu().numpy().tolist()
        pred_end_logits += b_end_logits
        b_ansTyp_logits = ansTyp_logits.detach().cpu().numpy().tolist()
        pred_ansTyp_logits += b_ansTyp_logits
        b_suppFacts_logits = suppFacts_logits.detach().cpu().numpy().tolist()
        pred_suppFacts_logits += b_suppFacts_logits

    pred_start_logits = np.asarray(pred_start_logits)
    pred_end_logits = np.asarray(pred_end_logits)
    pred_ansTyp_logits = np.asarray(pred_ansTyp_logits)
    pred_suppFacts_logits = np.asarray(pred_suppFacts_logits)

    np.save(args.predictions_save_path + "pred_start_logits.npy", pred_start_logits)
    np.save(args.predictions_save_path + "pred_end_logits.npy", pred_end_logits)
    np.save(args.predictions_save_path + "pred_ansTyp_logits.npy", pred_ansTyp_logits)
    np.save(args.predictions_save_path + "pred_suppFacts_logits.npy", pred_suppFacts_logits)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)