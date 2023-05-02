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


def evaluation(args, encoder, classifier, eval_dataloader):
    encoder.eval()
    classifier.eval()

    eval_preds = []
    eval_labels = []
    eval_losses = []
    
    for _, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        
        input_ids, attention_mask, labels = batch
        with torch.no_grad():        
            z = encoder(input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                    )

            sequence_output = z[0][:,0,:]
            outputs = classifier(sequence_output)
                
        eval_preds += torch.argmax(outputs, dim=1).cpu().numpy().tolist()
        eval_labels += labels.cpu().numpy().tolist()

    final_acc = np.mean(np.array(eval_preds)==np.array(eval_labels))
    final_f1 = f1_score(eval_labels, eval_preds)
    final_precision = precision_score(eval_labels, eval_preds)
    final_recall = recall_score(eval_labels, eval_preds)
    
    return final_acc, final_f1, final_precision, final_recall

def main(args):
    fix_random_seed_as(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_root = 'trained_models/baseline_models'
    save_dir = os.path.join(save_root, args.source_data_type)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.lm_model,
        do_lower_case=args.do_lower_case
        )

    encoder = CustomizedRobertaEncoder.from_pretrained(
        args.lm_model,
        output_attentions=False, 
        output_hidden_states=False
        )
    
    classifier = AnswerClassifier()

    train_dataloader = get_loader(args, mode='source_train', tokenizer=tokenizer)
    val_dataloader = get_loader(args, mode='source_val', tokenizer=tokenizer)
    test_dataloader = get_loader(args, mode='source_test', tokenizer=tokenizer)
    t_total = len(train_dataloader) * args.num_train_epochs

    encoder.resize_token_embeddings(len(tokenizer))
    
    encoder.to(args.device)
    classifier.to(args.device)

    optimizer_encoder = AdamW(encoder.parameters(), lr=args.learning_rate)
    optimizer_classifier = AdamW(classifier.parameters(), lr=args.learning_rate)
    
    global_step = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0
    best_f1 = 0
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            encoder.train()
            classifier.train()

            batch = tuple(t.to(args.device) for t in batch)
            
            input_ids, attention_mask, labels = batch
            z = encoder(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
                        
            sequence_output = z[0][:,0,:]
            logits = classifier(sequence_output)

            loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=-1)
            acc = torch.mean((predictions == labels).float())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)

            optimizer_encoder.step()
            optimizer_classifier.step()
            
            encoder.zero_grad()
            classifier.zero_grad()

            if step % 100 == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.3f}\tAcc: {:.3f}'.format(
                    epoch, step, len(train_dataloader), loss.item(), acc.item()))

            global_step += 1

        eval_acc, eval_f1, _, _ = evaluation(args, encoder, classifier, val_dataloader)
        print('Testing: Eval ACC: {:.3f}\teval_f1: {:.3f}\n'.format(eval_acc, eval_f1))

        if eval_acc + eval_f1 >= best_acc + best_f1:
            best_acc = eval_acc
            best_f1 = eval_f1
            print('***** Saving best model *****')

            encoder.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            classifier_save_name = os.path.join(save_dir, 'answer_classifier.checkpoint')
            classifier_checkpoint = {'state_dict': classifier.state_dict()}
            torch.save(classifier_checkpoint, classifier_save_name)
        


if __name__ == '__main__': 
    main(args)