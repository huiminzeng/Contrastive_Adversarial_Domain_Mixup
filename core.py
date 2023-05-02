import pdb

import numpy as np
import torch
import torch.nn.functional as F

def filter_predictions(logits):
    batch_size = logits.shape[0]
    prob = F.softmax(logits, dim=-1)

    max_prob, pseudo_label = torch.max(prob, dim=-1)
    batch_entry_idx = torch.where(max_prob > 0.8)[0]

    return pseudo_label, batch_entry_idx

def rbf(source_feature, target_feature, kernel_sigma_start=2, fix_sigma=None):
    batch_size = source_feature.shape[0]

    rbf_kernel_list = []
    for i in range(5):
        kernel_sigma = kernel_sigma_start ** i
        rbf_kernel = - torch.sum((source_feature - target_feature) ** 2, dim=-1) / (2 * (kernel_sigma ** 2))
        rbf_kernel = torch.mean(torch.exp(rbf_kernel))

        rbf_kernel_list.append(rbf_kernel)

    rbf_kernel = sum(rbf_kernel_list)

    return rbf_kernel

def source_sequence_adv(discriminator, target_feature, source_feature, epsilon):
    device = target_feature.device
    batch_size = target_feature.shape[0]

    criterion_kl = torch.nn.KLDivLoss(reduction='none')
    criterion_ce = torch.nn.CrossEntropyLoss()

    batch_size = len(target_feature)

    discriminator.eval()
    
    feature_all = torch.cat([target_feature, source_feature], dim=0)
    feature_adv = feature_all.clone().detach() + 0.001 * torch.randn(feature_all.shape).to(device)
    
    source_domain_label = torch.zeros(batch_size).to(device)
    target_domain_label = torch.ones(batch_size).to(device)
    label_all = torch.cat([target_domain_label, source_domain_label], dim=0).long()

    for _ in range(5):
        feature_adv.requires_grad_()
        with torch.enable_grad():            
            outputs_adv = discriminator(feature_adv)

            loss = criterion_ce(outputs_adv, label_all)
            
            predictions = torch.argmax(outputs_adv, dim=-1)
            acc = torch.mean((predictions == label_all).float())
            # print("acc: ", acc)
            grad = torch.autograd.grad(loss, [feature_adv])[0]
            
            feature_adv = feature_adv.detach() + epsilon / 5 * torch.sign(grad.detach())
    
    # pdb.set_trace()
    target_perturbation = (feature_adv-feature_all)[:batch_size].clone().detach()
    source_perturbation = (feature_adv-feature_all)[batch_size:].clone().detach()

    return source_perturbation, target_perturbation

def rbf_loss(discriminator, classifier,
            target_logits, source_logits,
            target_sequence_output, source_sequence_output,
            target_input_ids, target_attention_mask, target_labels,
            source_input_ids, source_attention_mask, source_labels,
            alpha, epsilon):

    target_perturbation, source_perturbation = source_sequence_adv(discriminator, target_sequence_output, source_sequence_output, epsilon)

    target_sequence_output = target_sequence_output + target_perturbation
    source_sequence_output = source_sequence_output + source_perturbation
    
    batch_size = source_input_ids.shape[0]
    criterion = torch.nn.CrossEntropyLoss()
    
    # getting pseudo label for target data
    target_labels_pseudo, batch_entry_idx = filter_predictions(target_logits)

    target_sequence_output = target_sequence_output[batch_entry_idx]
    target_labels = target_labels[batch_entry_idx]
    target_labels_pseudo = target_labels_pseudo[batch_entry_idx]

    loss_reg = 0
    if len(batch_entry_idx) != 0:
        source_pos_output = source_sequence_output[(source_labels==1).nonzero()]
        source_neg_output = source_sequence_output[(source_labels==0).nonzero()]
        source_pos_num = torch.sum(source_labels==1)
        source_neg_num = torch.sum(source_labels==0)

        target_pos_output = target_sequence_output[(target_labels_pseudo==1).nonzero()]
        target_neg_output = target_sequence_output[(target_labels_pseudo==0).nonzero()]
        target_pos_num = torch.sum(target_labels_pseudo==1)
        target_neg_num = torch.sum(target_labels_pseudo==0)
        
        # pdb.set_trace()
        if len(source_neg_output) > 0 and len(target_neg_output) > 0:
            num_selected = torch.min(source_neg_num, target_neg_num)
            loss_reg -= alpha * rbf(source_neg_output[:num_selected], target_neg_output[:num_selected])

        if len(source_pos_output) > 0 and len(target_pos_output) > 0:
            num_selected = torch.min(source_pos_num, target_pos_num)
            loss_reg -= alpha * rbf(source_pos_output[:num_selected], target_pos_output[:num_selected])

        if len(source_pos_output) > 0 and len(source_neg_output) > 0:
            num_selected = torch.min(source_pos_num, source_neg_num)
            loss_reg = alpha * rbf(source_pos_output[:num_selected], source_neg_output[:num_selected])
        
        if len(target_pos_output) > 0 and len(target_neg_output) > 0:
            num_selected = torch.min(target_pos_num, target_neg_num)
            loss_reg = alpha * rbf(target_pos_output[:num_selected], target_neg_output[:num_selected])

    target_logits = classifier(target_sequence_output)
    source_logits = classifier(source_sequence_output)

    loss_normal = criterion(source_logits, source_labels)
    loss_normal += criterion(target_logits, target_labels_pseudo)

    target_acc = torch.mean((torch.argmax(target_logits, dim=-1) == target_labels).float())
    return loss_normal, loss_reg, target_acc