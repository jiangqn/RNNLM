import torch
from src.utils.constants import PAD_INDEX
import math

def eval(model, data_loader, criterion):
    total_tokens = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            src, trg = data
            src, trg = src.cuda(), trg.cuda()
            batch_tokens = trg.size(0) * trg.size(1)
            logit = model(src)
            logit = logit.view(batch_tokens, -1)
            trg = trg.view(batch_tokens)
            loss = criterion(logit, trg)
            valid_tokens = (trg != PAD_INDEX).long().sum().item()
            total_tokens += valid_tokens
            total_loss += loss * valid_tokens
    loss = total_loss / total_tokens
    ppl = math.exp(loss)
    return loss, ppl