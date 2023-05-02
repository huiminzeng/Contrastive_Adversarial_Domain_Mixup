import json
import os
import argparse
import numpy as np
import random
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
from datetime import datetime

from dataset import get_dataset
from dataloader import preprocess, tokenize, get_loader

from models import CustomizedRobertaEncoder, DomainDiscriminator, AnswerClassifier

from core import rbf_loss

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
parser.add_argument("--epsilon", default=0.1, type=float)

parser.add_argument("--eval_mode", default='baselines', type=str)
# Evaluation mode
args = parser.parse_args()


def post_processing(discriminator, feature, epsilon):
    device = feature.device
    batch_size = feature.shape[0]
    criterion_ce = torch.nn.CrossEntropyLoss()
    
    feature_adv = feature.clone().detach() + 0.001 * torch.randn(feature.shape).to(device)
    domain_label = torch.ones(batch_size).to(device).long()

    for _ in range(5):
        feature_adv.requires_grad_()
        with torch.enable_grad():            
            outputs_adv = discriminator(feature_adv)
            loss = criterion_ce(outputs_adv, domain_label)
            
            predictions = torch.argmax(outputs_adv, dim=-1)
            acc = torch.mean((predictions == domain_label).float())
            
            grad = torch.autograd.grad(loss, [feature_adv])[0]
            
            feature_adv = feature_adv.detach() + epsilon / 10 * torch.sign(grad.detach())
    
    return feature_adv

def main(args):
    print(args)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("we are evaluating target domain: ", args.target_data_type)
    if args.eval_mode == 'baselines':
        load_root = 'trained_models/baseline_models'
        load_dir = os.path.join(load_root, args.source_data_type)
    
    elif args.eval_mode == 'adapted_rbf':
        load_root = 'trained_models/adapted_rbf'
        load_dir = os.path.join(load_root, 'source_domain_' + args.source_data_type, 'taget_domain_' + args.target_data_type, 'alpha_' + str(args.alpha), 'epsilon_' + str(args.epsilon))

    print("We are loading baseline models from: ", load_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        load_dir,
        do_lower_case=args.do_lower_case
        )

    encoder = CustomizedRobertaEncoder.from_pretrained(
        load_dir,
        output_attentions=False, 
        output_hidden_states=False
        )
    
    encoder.resize_token_embeddings(len(tokenizer))

    classifier = AnswerClassifier()
    classifier_load_name = os.path.join(load_dir, 'answer_classifier.checkpoint')
    classifier_checkpoint = torch.load(classifier_load_name)
    classifier.load_state_dict(classifier_checkpoint['state_dict'])

    if args.eval_mode == 'adapted_rbf':
        discriminator = DomainDiscriminator()
        discriminator_load_name = os.path.join(load_dir, 'domain_discriminator.checkpoint')
        discriminator_checkpoint = torch.load(discriminator_load_name)
        discriminator.load_state_dict(discriminator_checkpoint['state_dict'])

    target_test_dataloader = get_loader(args, mode='target_test', tokenizer=tokenizer)

    encoder.to(args.device)
    classifier.to(args.device)
    encoder.eval()
    classifier.eval()

    if args.eval_mode == 'adapted_rbf':
        discriminator.to(args.device)
        discriminator.eval()

    criterion = torch.nn.CrossEntropyLoss()
    eval_preds = []
    eval_labels = []
    for step, target_batch in enumerate(target_test_dataloader):    
        target_batch = tuple(t.to(args.device) for t in target_batch)
        target_input_ids, target_attention_mask, target_labels = target_batch

        with torch.no_grad():   
            target_z = encoder(input_ids=target_input_ids,
                        attention_mask=target_attention_mask,
                        labels=target_labels)

            target_sequence_output = target_z[0][:,0,:]

            if args.eval_mode == 'adapted_rbf':
                target_sequence_output = post_processing(discriminator, target_sequence_output, args.epsilon)
            target_logits = classifier(target_sequence_output)
                    
            eval_preds += torch.argmax(target_logits, dim=1).cpu().numpy().tolist()
            eval_labels += target_labels.cpu().numpy().tolist()

    final_bacc = balanced_accuracy_score(eval_labels, eval_preds)
    final_acc = np.mean(np.array(eval_preds)==np.array(eval_labels))
    final_f1 = f1_score(eval_labels, eval_preds)
    final_precision = precision_score(eval_labels, eval_preds)
    final_recall = recall_score(eval_labels, eval_preds)

    print('Test Over!!  BACC: {:.4f};  ACC: {:.4f};  F1: {:.4f}'.format(
                final_bacc, final_acc, final_f1))
    pdb.set_trace()
if __name__ == '__main__': 
    main(args)