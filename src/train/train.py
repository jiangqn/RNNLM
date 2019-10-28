import os
import yaml
import math
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
from src.language_model.rnnlm import RNNLM
from src.data_process.dataset import LMDataset
from src.utils.constants import PAD_INDEX
from src.utils.logger import Logger
from src.train.eval import eval

def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    base_path = os.path.join('./data', args.data)
    processed_base_path = os.path.join(base_path, 'processed')
    processed_train_path = os.path.join(processed_base_path, 'train.npz')
    processed_valid_path = os.path.join(processed_base_path, 'valid.npz')
    glove_path = os.path.join(processed_base_path, 'glove.npy')
    log_base_path = os.path.join(base_path, 'log')
    log_path = os.path.join(log_base_path, 'train_log.txt')
    data_log_path = os.path.join(log_base_path, 'data_log.yml')
    data_log = yaml.safe_load(open(data_log_path, 'r'))
    vocab_size = data_log['vocab_size']
    logger = Logger(log_path)
    logger.log('make data')
    train_data = LMDataset(processed_train_path)
    valid_data = LMDataset(processed_valid_path)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    logger.log('make model')
    model = RNNLM(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    logger.log('load pretrained embeddings')
    model.load_pretrained_embeddings(glove_path, fixed=args.embedding_fixed=='True')
    logger.log('transfer model to gpu')
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    logger.log('train start')
    for epoch in range(args.epoches):
        total_tokens = 0
        total_loss = 0
        for i, data in enumerate(train_loader):
            model = model.train()
            optimizer.zero_grad()
            src, trg = data
            src, trg = src.cuda(), trg.cuda()
            logit = model(src)
            logit = logit.view(-1, vocab_size)
            trg = trg.view(-1)
            loss = criterion(logit, trg)
            loss.backward()
            optimizer.step()
            valid_tokens = (trg != PAD_INDEX).long().sum().item()
            total_tokens += valid_tokens
            total_loss += loss * valid_tokens
            if i % 100 == 0:
                train_loss = total_loss / total_tokens
                train_ppl = math.exp(train_loss)
                total_loss, total_tokens = 0, 0
                val_loss, val_ppl = eval(model, valid_loader, criterion)
                logger.log('[epoch %d step %4d] train_loss: %.4f\ttrain_ppl: %.4f\tval_loss: %.4f\tval_ppl: %.4f' %
                           (epoch, i, train_loss, train_ppl, val_loss, val_ppl))