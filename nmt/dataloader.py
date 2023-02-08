import torch
from torch.utils.data import DataLoader, TensorDataset
import regex as re
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

DATASET_NAME = "cnn_dailymail"
CONFIG = "3.0.0"
TRANSLATION_PREFIX = ">>nob<< "

SPLIT_PATTERN = "(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
EMB_SIZE = 512
NHEAD = 8
BATCH_SIZE = 128

dataset= load_dataset(DATASET_NAME, CONFIG, split='test')
def clean_up_example(example):
    #Pre-processing
    # Append >>nob<< token
    example["article"] = [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(SPLIT_PATTERN, example["article"])] + ["<EOA>"]#.append("<article-end>")
    example["highlights"] = [TRANSLATION_PREFIX + sentence.strip() for sentence in re.split(SPLIT_PATTERN, example["highlights"])] + ["<EOA>"] #.append("<highlights-end>")
    return (example["article"], example["highlights"], example["id"])

dataset = dataset.map(clean_up_example, batched=True, batch_size=1000, remove_columns=["article", "highlights", "id"],)
dataset = 

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

#inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
#tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
#dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True, num_workers=2, prefetch_factor=2)
for batch_ndx, sample in enumerate(loader):
    p

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())