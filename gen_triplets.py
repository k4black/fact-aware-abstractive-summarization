from tqdm.auto import tqdm

from dataset import CnnDmDataset
from torch.utils.data import DataLoader

dataset_train = CnnDmDataset('train')
dataset_val = CnnDmDataset('validation')
dataset_test = CnnDmDataset('test')

print()

BATCH_SIZE = 2

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=40, collate_fn=CnnDmDataset.collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=40, collate_fn=CnnDmDataset.collate_fn)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=40, collate_fn=CnnDmDataset.collate_fn)

for i in tqdm(dataloader_train):
    i
    pass