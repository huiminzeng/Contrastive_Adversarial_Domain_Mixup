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
from sklearn.metrics import f1_score, precision_score, recall_score
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

args = parser.parse_args()

def main(args):
    fix_random_seed_as(args.seed)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    load_root = 'trained_models/baseline_models'
    load_dir = os.path.join(load_root, args.source_data_type)
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

    load_discriminator_root = 'trained_models/baseline_discriminators'
    load_discriminator_dir = os.path.join(load_discriminator_root, 'source_domain_' + args.source_data_type, 'taget_domain_' + args.target_data_type)
    print("We are loading baseline discriminators from: ", load_dir)
    discriminator = DomainDiscriminator()
    discriminator_load_name = os.path.join(load_discriminator_dir, 'domain_discriminator.checkpoint')
    discriminator_checkpoint = torch.load(discriminator_load_name)
    discriminator.load_state_dict(discriminator_checkpoint['state_dict'])

    print("All models loaded!!!")

    save_root = 'trained_models/adapted_rbf'
    save_dir = os.path.join(save_root, 'source_domain_' + args.source_data_type, 'taget_domain_' + args.target_data_type, 'alpha_' + str(args.alpha), 'epsilon_' + str(args.epsilon))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("We are saving adapted models to: ", save_dir)
    target_train_dataloader = get_loader(args, mode='target_train', tokenizer=tokenizer)
    target_val_dataloader = get_loader(args, mode='target_val', tokenizer=tokenizer)
    target_test_dataloader = get_loader(args, mode='target_test', tokenizer=tokenizer)

    # process source domain data sampler
    source_pointer = [0] * 2
    source_label_dict = {0: [], 1: []}
    data, labels = get_dataset(args, 'train', args.source_data_type, args.source_data_path).load_dataset()
    for idx, label in enumerate(labels):
        source_label_dict[label].append(idx)
    for key in source_label_dict.keys():
        random.shuffle(source_label_dict[key])

    inputs = preprocess(args, data)
    all_input_ids, all_token_type_ids, all_attention_mask = tokenize(args, inputs, tokenizer)
    source_dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, torch.tensor(labels))

    t_total = len(target_train_dataloader) * args.num_train_epochs
    
    encoder.to(args.device)
    classifier.to(args.device)
    discriminator.to(args.device)

    optimizer_encoder = AdamW(encoder.parameters(), lr=args.learning_rate)
    optimizer_classifier = AdamW(classifier.parameters(), lr=args.learning_rate)
    optimizer_discriminator = AdamW(discriminator.parameters(), lr=args.learning_rate)

    global_step = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0
    best_f1 = 0
    for epoch in range(args.num_train_epochs):
        #######################################################################
        ###################   domain adaptation training   ####################
        #######################################################################
        for step, target_batch in enumerate(target_train_dataloader):
            encoder.train()
            classifier.train()
            
            target_batch = tuple(t.to(args.device) for t in target_batch)
            target_input_ids, target_attention_mask, target_labels = target_batch
            batch_size = target_input_ids.shape[0]
            target_domain_label = torch.ones(batch_size).to(args.device)

            target_z = encoder(input_ids=target_input_ids,
                        attention_mask=target_attention_mask,
                        labels=target_labels)
            target_sequence_output = target_z[0][:,0,:]
            target_logits = classifier(target_sequence_output)

            # sampling source batch with same labels
            source_batch = sample_source_batch(target_labels, source_dataset, source_label_dict, source_pointer)

            source_batch = tuple(t.to(args.device) for t in source_batch)
            source_input_ids, source_attention_mask, source_labels = source_batch
            batch_size = source_input_ids.shape[0]
            source_domain_label = torch.zeros(batch_size).to(args.device)

            source_z = encoder(input_ids=source_input_ids,
                        attention_mask=source_attention_mask,
                        labels=source_labels)
            source_sequence_output = source_z[0][:,0,:]
            source_logits = classifier(source_sequence_output)
            
            loss_normal, loss_reg, target_acc = rbf_loss(discriminator, classifier,
                                                        target_logits, source_logits,
                                                        target_sequence_output, source_sequence_output,
                                                        target_input_ids, target_attention_mask, target_labels,
                                                        source_input_ids, source_attention_mask, source_labels,
                                                        args.alpha, args.epsilon)

            loss = loss_normal + loss_reg
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)

            optimizer_encoder.step()
            optimizer_classifier.step()
            
            encoder.zero_grad()
            classifier.zero_grad()

            #######################################################################
            ###################   domain discriminator training   #################
            #######################################################################
            target_z = encoder(input_ids=target_input_ids,
                        attention_mask=target_attention_mask,
                        labels=target_labels)
            target_sequence_output = target_z[0][:,0,:]
            target_outputs = discriminator(target_sequence_output)

            source_z = encoder(input_ids=source_input_ids,
                        attention_mask=source_attention_mask,
                        labels=source_labels)
            source_sequence_output = source_z[0][:,0,:]
            source_outputs = discriminator(source_sequence_output)

            outputs_all = torch.cat([target_outputs, source_outputs], dim=0)
            label_all = torch.cat([target_domain_label, source_domain_label], dim=0).long()

            loss_adv = criterion(outputs_all, label_all)
            loss_adv.backward()

            predictions = torch.argmax(outputs_all, dim=-1)

            acc = torch.mean((predictions == label_all).float())
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)

            optimizer_discriminator.step()

            optimizer_encoder.zero_grad()
            optimizer_discriminator.zero_grad()

            if step % 100 == 0:
                print('Train Epoch: {} [{}/{}]  Loss: {:.3f};  Reg: {:.3f};  Acc: {:.3f};  Dis Acc: {:.3f}'.format(
                    epoch, step, len(target_train_dataloader), loss_normal.item(), loss_reg.item(), target_acc.item(), acc.item()))

        eval_acc, eval_f1, _, _ = evaluation(args, encoder, classifier, target_val_dataloader)
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

            discriminator_save_name = os.path.join(save_dir, 'domain_discriminator.checkpoint')
            discriminator_checkpoint = {'state_dict': discriminator.state_dict()}
            torch.save(discriminator_checkpoint, discriminator_save_name)

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

def sample_source_batch(target_labels, source_dataset, source_label_dict, source_pointer):
        output_idx = []
        for label in target_labels.tolist():
            next_idx = source_pointer[label] % len(source_label_dict[label])
            output_idx.append(source_label_dict[label][next_idx])
            source_pointer[label] += 1

        all_input_ids, all_token_type_ids, all_attention_mask, labels  = [], [], [], []
        for idx in output_idx:
            all_input_ids.append(source_dataset[idx][0].unsqueeze(0))
            
            all_attention_mask.append(source_dataset[idx][-2].unsqueeze(0))
            labels.append(source_dataset[idx][-1].unsqueeze(0))
        
        all_input_ids = torch.vstack(all_input_ids)
        
        all_attention_mask = torch.vstack(all_attention_mask)
        labels = torch.vstack(labels).squeeze()

        return all_input_ids, all_attention_mask, labels
            

if __name__ == '__main__': 
    main(args)