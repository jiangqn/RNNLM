import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='train', choices=['preprocess', 'train', 'test'])
parser.add_argument('--data', type=str, default='ptb', choices=['ptb'])
parser.add_argument('--embed_size', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=32, choices=[16, 32, 64, 128])
parser.add_argument('--epoches', type=int, default=30)
parser.add_argument('--glove_source_path', type=str, default='../../datasets/embeddings/glove.840B.300d.txt')
parser.add_argument('--embedding_fixed', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--gpu', type=int, default=2, choices=[0, 1, 2, 3])

args = parser.parse_args()

if args.task == 'preprocess':
    from src.data_process.preprocess import preprocess
    preprocess(args)
elif args.task == 'train':
    from src.train.train import train
    train(args)
elif args.task == 'test':
    from src.train.test import test
    test(args)
else:
    raise ValueError('argument --task %s is invalid.' % args.task)