from src.utils.constants import PAD_INDEX

def sentence_clip(sentence):
    mask = sentence != PAD_INDEX
    lens = mask.long().sum(dim=1, keepdim=False)
    max_len = lens.max().item()
    sentence = sentence[:, 0: max_len].contiguous()
    return sentence