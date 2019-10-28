import os
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from src.data_process.dataset import LMDataset
from src.train.eval import eval
from src.utils.constants import PAD_INDEX
from src.utils.logger import Logger

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    base_path = os.path.join('./data', args.data)
    processed_base_path = os.path.join(base_path, 'processed')
    processed_test_path = os.path.join(processed_base_path, 'test.npz')
    save_path = os.path.join(processed_base_path, 'rnnlm.pkl')
    log_base_path = os.path.join(base_path, 'log')
    log_path = os.path.join(log_base_path, 'test_log.txt')
    logger = Logger(log_path)
    test_data = LMDataset(processed_test_path)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    model = torch.load(save_path)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    test_loss, test_ppl = eval(model, test_loader, criterion)
    logger.log('test_loss: %.4f\ttest_ppl: %.4f' % (test_loss, test_ppl))