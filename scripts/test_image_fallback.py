import sys, os
# ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from diffuvqa.vqa_datasets import get_corpus, ImageDataset
from basic_utils import load_tokenizer
class A: pass
args = A()
args.dataset = 'qqp'
args.data_dir = r'C:\Users\BARIS\Documents\GitHub\DiffuVQA\datasets'
args.vocab = 'bert'
args.checkpoint_path = 'diffuvqa/config'
args.image_size = 224
print('loading corpus...')
tokenizer = load_tokenizer(args)
train_dataset, data_lst = get_corpus(args, seq_len=128, split='train', loaded_vocab=tokenizer)
print('creating image dataset...')
img_ds = ImageDataset(args.data_dir, data_lst, args)
img = img_ds[0]
print('got image object:', type(img))
try:
    # if it's a tensor, print shape
    print('shape/size:', getattr(img, 'shape', getattr(img, 'size', 'n/a')))
except Exception as e:
    print('shape probe failed:', e)
print('done')
