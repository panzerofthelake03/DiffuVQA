import sys, os
# ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
	sys.path.insert(0, repo_root)
from diffuvqa.vqa_datasets import get_corpus
class A: pass
args = A()
args.dataset = 'qqp'
args.data_dir = r'C:\Users\BARIS\Documents\GitHub\DiffuVQA\datasets'
# Prepare a tokenizer instance (loaded_vocab) so helper_tokenize has a tokenizer
from basic_utils import load_tokenizer
args.vocab = 'bert'
args.checkpoint_path = 'diffuvqa/config'
print('attempting to load corpus...')
tokenizer = load_tokenizer(args)
train_dataset, data_lst = get_corpus(args, seq_len=128, split='train', loaded_vocab=tokenizer)
print('Loaded corpus OK')
print('sample questions:', data_lst['question'][:2])
print('sample answers:', data_lst['answer'][:2])
