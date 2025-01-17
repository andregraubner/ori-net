import random
import torch
import torch.nn.functional as F

def jaccard_loss(preds, true, eps=1e-7):
    
    true_1_hot = F.one_hot(true)
    true_1_hot = true_1_hot.type(preds.type())
    
    intersection = torch.sum(preds * true_1_hot, dim=0)
    cardinality = torch.sum(preds + true_1_hot, dim=0)
    
    union = cardinality - intersection
    
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)

def shift_ori(s, start, end, max_length):
    
    # Shift the string so that the start index is at the front
    shifted_string = s[start:] + s[:start]

    # New indices for start and end
    new_start_index = 0
    new_end_index = (end - start) % len(s)

    # Randomly shift ORI inside window
    shift_amount = random.randint(0, min(len(s) - new_end_index - 1, max_length-new_end_index-1))
    shifted_string = shifted_string[-shift_amount:] + shifted_string[:-shift_amount]
    shifted_string = shifted_string[:max_length]

    start = shift_amount
    end = new_end_index + shift_amount
    
    return shifted_string, start, end