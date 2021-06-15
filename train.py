#! /usr/bin/env python

import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from transformers import ElectraTokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

from models import ElectraQA

MAXLEN = 512

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=3e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=1, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')
parser.add_argument('--data_path', type=str, help='Load path to training data')

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

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    with open(args.data_path) as f:
        all_samples = json.load(f)

    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)

    start_positions_true = []
    end_positions_true = []
    supporting_facts_start_positions_vector_true = []
    answer_type_true = []   # 0 indicates 'yes'; 1 indicates 'no'; 2 indicates span
    input_ids = []
    attention_masks = []
    token_type_ids = []

    count = 0
    print(len(all_samples))
    for sample in all_samples:
        count+=1
        if count%100==0:
            print(count)
        question = sample["question"]
        context = "".join(sample["context"][0][1]) + " " + "".join(sample["context"][1][1])
        combo = question + " [SEP] " + context
        input_encodings_dict = tokenizer(combo, truncation=True, max_length=MAXLEN, padding="max_length")
        inp_ids = input_encodings_dict['input_ids']
        inp_att_msk = input_encodings_dict['attention_mask']
        tok_type_ids = [0 if i<= inp_ids.index(102) else 1 for i in range(len(inp_ids))]  # Indicates whether part of sentence A or B -> 102 is Id of [SEP] token

        answer = sample['answer']
        if answer == 'yes':
            ans_typ = 0
            start_idx, end_idx = 0, 0
        elif answer == 'no':
            ans_typ = 1
            start_idx, end_idx = 0, 0
        else:
            ans_typ = 2
            ans_ids = tokenizer.encode(answer)[1:-1]
            start_idx, end_idx = _find_sub_list(ans_ids, inp_ids)
            if start_idx == -1:
                print("Didn't find answer")
                print(answer)
                print(context)
                continue

        # Get positions of supporting facts with 1s at all positions with the sentence start corresponding to a supporting fact
        supp_start_idxs = []
        for supp in sample['supporting_facts']:
            title = supp[0]
            sentence_num = supp[1]
            if title == sample["context"][0][0]:
                if sentence_num >= len(sample["context"][0][1]):
                    print("Sentence does not exist in context")
                    continue
                sentence = sample["context"][0][1][sentence_num]
            else:
                assert title == sample["context"][1][0]
                if sentence_num >= len(sample["context"][1][1]):
                    print("Sentence does not exist in context")
                    continue
                sentence = sample["context"][1][1][sentence_num]
            supp_ids = tokenizer.encode(sentence)[1:-1]
            start_pos, _ = _find_sub_list(supp_ids, inp_ids)
            if start_idx == -1:
                print("Didn't find supporting fact")
                print(sentence)
                print(context)
                continue
            supp_start_idxs.append(start_pos)
        # Build the vector
        if -1 in supp_start_idxs:
            continue
        supp_start_vec = [1 if i in supp_start_idxs else 0 for i in range(len(inp_ids))]

        start_positions_true.append(start_idx)
        end_positions_true.append(end_idx)
        supporting_facts_start_positions_vector_true.append(supp_start_vec)
        answer_type_true.append(ans_typ)
        input_ids.append(inp_ids)
        token_type_ids.append(tok_type_ids)
        attention_masks.append(inp_att_msk)

    start_positions_true = torch.tensor(start_positions_true)
    start_positions_true = start_positions_true.long().to(device)
    end_positions_true = torch.tensor(end_positions_true)
    end_positions_true = end_positions_true.long().to(device)
    supporting_facts_start_positions_vector_true = torch.tensor(supporting_facts_start_positions_vector_true)
    supporting_facts_start_positions_vector_true = supporting_facts_start_positions_vector_true.long().to(device)
    answer_type_true = torch.tensor(answer_type_true)
    answer_type_true = answer_type_true.long().to(device)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    token_type_ids = torch.tensor(token_type_ids)
    token_type_ids = token_type_ids.long().to(device)
    attention_masks = torch.tensor(attention_masks)
    attention_masks = attention_masks.long().to(device)


    train_data = TensorDataset(start_positions_true, end_positions_true, supporting_facts_start_positions_vector_true, answer_type_true, input_ids, token_type_ids, attention_masks)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    model = ElectraQA().to(device)

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    # weight_decay = 0.01
                    )

    total_steps = len(train_dataloader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 306,
                                                num_training_steps = total_steps)

    criterion_qa = torch.nn.CrossEntropyLoss()
    criterion_ansTyp = torch.nn.CrossEntropyLoss()
    criterion_suppFacts = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.n_epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0
        model.train()
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_start_pos_true = batch[0].to(device)
            b_end_pos_true = batch[1].to(device)
            b_supp_facts_vec = batch[2].to(device)
            b_ans_typ = batch[3].to(device)
            b_input_ids = batch[4].to(device)
            b_tok_typ_ids = batch[5].to(device)
            b_att_msks = batch[6].to(device)
            model.zero_grad()

            start_logits, end_logits, ansTyp_logits, suppFacts_logits = model(input_ids=b_input_ids, attention_mask=b_att_msks, token_type_ids=b_tok_typ_ids)

            loss_start = criterion_qa(start_logits, b_start_pos_true)
            loss_end = criterion_qa(end_logits, b_end_pos_true)
            loss_qa = (loss_start + loss_end) / 2
            loss_ansTyp = criterion_ansTyp(ansTyp_logits, b_ans_typ)
            loss_suppFacts = criterion_suppFacts(suppFacts_logits, b_supp_facts_vec)
            loss = loss_qa + loss_ansTyp + loss_suppFacts
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # Save the model to a file
    file_path = args.save_path+'electra_multi_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)