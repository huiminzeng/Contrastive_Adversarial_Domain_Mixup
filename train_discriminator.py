import json
import os
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime

from dataset import get_dataset
from dataloader import preprocess, tokenize, get_loader

from models import CustomizedRobertaEncoder, DomainDiscriminator, AnswerClassifier

import pdb

parser = argparse.ArgumentParser()

# dataset and dataloader
parser.add_argument('--seed', default=0, type=int,
                    help="Random seed for initialization")
parser.add_argument('--device', default='cuda', type=str,
                    help="Device used for training")

# Model settings
parser.add_argument("--lm_model", default='roberta-base', type=str,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese")

parser.add_argument("--save_dir", default='trained_models', type=str,
                    help="The output directory where the model checkpoints and predictions will be written")
parser.add_argument("--max_seq_length", default=512, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                            "longer than this will be truncated, and sequences shorter than this will be padded")

# Experiment settings
parser.add_argument("--source_data_path", default='./data/LIAR', type=str, help="Source file for training. E.g., ./data/Constraint")
parser.add_argument("--source_data_type", default='liar', type=str, help="Source file type for training. E.g., constraint")
parser.add_argument("--target_data_path", default='./data/Constraint', type=str, help="Target file training for joint training")
parser.add_argument("--target_data_type", default='constraint', type=str, help="Source file type for training. E.g., constraint")

# Preprocessing settings
parser.add_argument("--do_lower_case", default=True, help="Whether to lower case the input text")  # applies to BERT
parser.add_argument("--separate_special_symbol", default=True, help="Whether to separate # and @ in input text")
parser.add_argument("--translate_emoji", default=True, help="Whether to translate emojis in input text")
parser.add_argument("--tokenize_url", default=True, help="Whether to tokenize urls in input text")
parser.add_argument("--tokenize_emoji", default=False, help="Whether to tokenize emojis in input text")
parser.add_argument("--tokenize_smiley", default=True, help="Whether to tokenize smileys in input text")
parser.add_argument("--tokenize_hashtag", default=True, help="Whether to tokenize hashtags in input text")
parser.add_argument("--tokenize_mention", default=True, help="Whether to tokenize mentions in input text")
parser.add_argument("--tokenize_number", default=False, help="Whether to tokenize numbers in input text")
parser.add_argument("--tokenize_reserved", default=False, help="Whether to tokenize reserved words in input text")
parser.add_argument("--remove_escape_char", default=True, help="Whether to remove escape characters in input text")
parser.add_argument("--remove_url", default=False, help="Whether to remove urls in input text")
parser.add_argument("--remove_emoji", default=False, help="Whether to remove emojis in input text")
parser.add_argument("--remove_smiley", default=False, help="Whether to remove smileys in input text")
parser.add_argument("--remove_hashtag", default=False, help="Whether to remove hashtags in input text")
parser.add_argument("--remove_mention", default=False, help="Whether to remove mentions in input text")
parser.add_argument("--remove_number", default=False, help="Whether to remove numbers in input text")
parser.add_argument("--remove_reserved", default=False, help="Whether to remove reserved words in input text")
parser.add_argument("--val_size", default=0.1, type=float, help="Validation size from the dataset if not already split")
parser.add_argument("--test_size", default=0.2, type=float, help="Test size from the dataset if not already split")

# Optimization settings
parser.add_argument("--train_batchsize", default=24, type=int, help="Batch size used for training")
parser.add_argument("--eval_batchsize", default=48, type=int, help="Batch size used for evaluation")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="The maximum norm for backward gradients")
parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 of training")

# Optimization settings
parser.add_argument("--alpha", default=0.01, type=float, help="Alpha for reversal gradient layer in DAT and CDA")
parser.add_argument("--conf_threshold", default=0.6, type=float, help="Alpha for reversal gradient layer in DAT and CDA")

args = parser.parse_args()

def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluation(args, encoder, discriminator, target_eval_dataloader, source_eval_dataloader_iter):
    encoder.eval()
    discriminator.eval()

    eval_preds = []
    eval_labels = []
    
    for _, batch in enumerate(target_eval_dataloader):
        with torch.no_grad():
            batch = tuple(t.cuda() for t in batch)
            target_input_ids, target_attention_mask, target_labels = batch
            batch_size = target_input_ids.shape[0]
            target_domain_label = torch.ones(batch_size).cuda()

            target_z = encoder(input_ids=target_input_ids,
                                attention_mask=target_attention_mask,
                                labels=target_labels)
            target_sequence_output = target_z[0][:,0,:]
            target_logits = discriminator(target_sequence_output)

            try:
                batch = next(source_eval_dataloader_iter)
            except:
                batch = batch
            
            batch = tuple(t.cuda() for t in batch)
            source_input_ids, source_attention_mask, source_labels = batch
            batch_size = source_input_ids.shape[0]
            source_domain_label = torch.zeros(batch_size).cuda()

            source_z = encoder(input_ids=source_input_ids,
                        attention_mask=source_attention_mask,
                        labels=source_labels)
            source_sequence_output = source_z[0][:,0,:]
            source_logits = discriminator(source_sequence_output)
            
            logits_all = torch.cat([target_logits, source_logits], dim=0)
            label_all = torch.cat([target_domain_label, source_domain_label], dim=0).long()

            eval_preds += torch.argmax(logits_all, dim=1).cpu().numpy().tolist()
            eval_labels += label_all.cpu().numpy().tolist()

            # print("preds: ", torch.argmax(logits_all, dim=1))
            # print("labels: ", label_all)

        # pdb.set_trace()

    final_acc = np.mean(np.array(eval_preds)==np.array(eval_labels))
    
    return final_acc

def main(args):
    print(args)
    fix_random_seed_as(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_root = 'trained_models/baseline_models'
    load_dir = os.path.join(load_root, args.source_data_type)
    print("we are loading encoder from: ", load_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        load_dir,
        do_lower_case=args.do_lower_case
        )

    encoder = CustomizedRobertaEncoder.from_pretrained(
        load_dir,
        output_attentions=False, 
        output_hidden_states=False
        )
    
    save_root = 'trained_models/baseline_discriminators'
    save_dir = os.path.join(save_root, 'source_domain_' + args.source_data_type, 'taget_domain_' + args.target_data_type)
    print("we are save the baseline discriminator to: ", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    discriminator = DomainDiscriminator()

    source_train_dataloader = get_loader(args, mode='source_train', tokenizer=tokenizer)
    source_val_dataloader = get_loader(args, mode='source_val', tokenizer=tokenizer)
    source_test_dataloader = get_loader(args, mode='source_test', tokenizer=tokenizer)

    target_train_dataloader = get_loader(args, mode='target_train', tokenizer=tokenizer)
    target_val_dataloader = get_loader(args, mode='target_val', tokenizer=tokenizer)
    target_test_dataloader = get_loader(args, mode='target_test', tokenizer=tokenizer)

    t_total = len(target_train_dataloader) * args.num_train_epochs

    encoder.resize_token_embeddings(len(tokenizer))
    
    encoder.to(args.device)
    discriminator.to(args.device)

    optimizer_discriminator = AdamW(discriminator.parameters(), lr=args.learning_rate)
    
    global_step = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0
    best_f1 = 0
    for epoch in range(args.num_train_epochs):
        source_train_dataloader_iter = iter(source_train_dataloader)
        source_val_dataloader_iter = iter(source_val_dataloader)

        for step, batch in enumerate(target_train_dataloader):
            encoder.train()
            discriminator.train()

            batch = tuple(t.to(args.device) for t in batch)
            target_input_ids, target_attention_mask, target_labels = batch
            batch_size = target_input_ids.shape[0]
            target_domain_label = torch.ones(batch_size).cuda()

            target_z = encoder(input_ids=target_input_ids,
                        attention_mask=target_attention_mask,
                        labels=target_labels)
            target_sequence_output = target_z[0][:,0,:]
            target_logits = discriminator(target_sequence_output)

            try:
                batch = next(source_train_dataloader_iter)
            except:
                batch = batch
            
            batch = tuple(t.cuda() for t in batch)
            source_input_ids, source_attention_mask, source_labels = batch
            batch_size = source_input_ids.shape[0]
            source_domain_label = torch.zeros(batch_size).cuda()

            source_z = encoder(input_ids=source_input_ids,
                        attention_mask=source_attention_mask,
                        labels=source_labels)
            source_sequence_output = source_z[0][:,0,:]
            source_logits = discriminator(source_sequence_output)
            
            logits_all = torch.cat([target_logits, source_logits], dim=0)
            label_all = torch.cat([target_domain_label, source_domain_label], dim=0).long()

            loss = criterion(logits_all, label_all)
            predictions = torch.argmax(logits_all, dim=-1)

            acc = torch.mean((predictions == label_all).float())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)

            optimizer_discriminator.step()
            
            discriminator.zero_grad()

            if step % 100 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.3f}\tAcc: {:.3f}'.format(
                    epoch, step, len(target_train_dataloader), loss.item(), acc.item()))

            global_step += 1

        eval_acc = evaluation(args, encoder, discriminator, target_val_dataloader, source_val_dataloader_iter)
        print('Testing: Eval ACC: {:.3f}\n'.format(eval_acc))

        if eval_acc >= best_acc:
            best_acc = eval_acc
            print('***** Saving best model *****')

            discriminator_save_name = os.path.join(save_dir, 'domain_discriminator.checkpoint')
            discriminator_checkpoint = {'state_dict': discriminator.state_dict()}
            torch.save(discriminator_checkpoint, discriminator_save_name)
        

if __name__ == '__main__': 
    main(args)