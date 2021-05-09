import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# torch.autograd.set_detect_anomaly(True)


from dataset import CnnDmDataset
from torch.utils.data import DataLoader


dataset_train = CnnDmDataset('train')
dataset_val = CnnDmDataset('validation')



BATCH_SIZE = 6  # no gat 2080ti
# BATCH_SIZE = 4  # gat 2080ti
# BATCH_SIZE = 14   # gat 3090
BATCH_SIZE_TEST = 20  # no gat 2080ti
# BATCH_SIZE_TEST = 15  # gat 2080ti
# BATCH_SIZE_TEST = 24  # gat 3090


dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=20, collate_fn=CnnDmDataset.collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=10, collate_fn=CnnDmDataset.collate_fn)



from models import get_model
GRAPH = False
tokenizer, model = get_model(graph=GRAPH, encoders=6, decoders=6, shared_head=True, pretrained=False)





import torch
import  torch.nn as nn
import  torch.nn.functional as F
from tqdm.auto import tqdm


from common import get_criterion_loss, _get_criterion_loss
import neptune.new as neptune

run = neptune.init(project='k4black/diploma-sum', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZDliYzEyZS1hOWVkLTQ1ZDQtOThlYS1jNDhhOTFjMGQ4ZjAifQ==')
run["sys/tags"].add(['pretrain'])

from transformers import AdamW, Adafactor, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

MAX_INPUT = 512
MAX_OUTPUT = 256
LR = 1e-5
# LR = 0.1
EPOCHS = 20
start_epoch = 0
DEVICE = 'cuda'
# DEVICE = 'cpu'


scaler = torch.cuda.amp.GradScaler()  # mixed precision

criterion = lambda x, y, m: get_criterion_loss(x, y, m)
optimizer = AdamW(model.parameters(), lr=LR)
# optimizer = Adafactor(model.parameters(), warmup_init=True)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1024, num_training_steps=1024*EPOCHS)
# scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=1024, num_training_steps=2*1024*EPOCHS, num_cycles=4)
# scheduler = None


CHECKPOINT = 'model-best-e6-g3-d6' if GRAPH else 'model-best-e6-g0-d6'
# CHECKPOINT = None
# CHECKPOINT = 'model-e_3-l_3.7768'
if CHECKPOINT is not None:
    print(f'LOADING checkpoint <{CHECKPOINT}>...')
    checkpoint = torch.load(f'pre-checkpoints/{CHECKPOINT}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss'] if 'loss' in checkpoint else None
    print(f'    saved loss: {loss}')

# optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=LR)
# optimizer = Adafactor(model.parameters(), warmup_init=True)
# optimizer = AdamW(model.parameters(), lr=LR)

start_epoch = 0

model = model.to(DEVICE)
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to(DEVICE)
# model = nn.DataParallel(model, device_ids=[1, 4, 5, 6]).to(DEVICE)
# model = nn.DataParallel(model).to(DEVICE)
# model.lm_head = model.lm_head.to('cuda:1')
# model.lm_head = model.lm_head.to('cuda:1')
# torch.distributed.init_process_group(backend='nccl', init_method='env://')
# model = nn.parallel.DistributedDataParallel(model, device_ids=[1, 3, 4, 5, 6]).to(DEVICE)


for param in model.parameters():
    param.requires_grad = True
# for param in model.shared.parameters():
#     param.requires_grad = False

from torchinfo import summary
summary(model, depth=3)


run["parameters"] = {
    "train/batchsize": BATCH_SIZE,
    "val/batchsize": BATCH_SIZE_TEST,
    "pretraining": model.config.pretrained,
    "max_input": MAX_INPUT,
    "max_output": MAX_OUTPUT,
    "learning_rate": LR,
    "criterion": "CrossEntropyLoss",
    "optimizer": type(optimizer).__name__,
    "scheduler": type(scheduler).__name__ if scheduler else None,
    "model": "pegasus+gat" if GRAPH else "pegasus",
    "encoders": model.config.encoder_layers,
    "gat": 3 if GRAPH else 0,
    "decoders": model.config.decoder_layers,
}


def checkpoint(model, optimizer, epoch, metrics=None):
    if metrics is not None:
        filename = f'pre-checkpoints/model-e_{epoch}-l_{metrics["loss"]:.4f}.pt'
    else:
        filename = f'pre-checkpoints/model-e_{epoch}.pt'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': metrics["loss"] if metrics else None,
    }, filename)


sent_mask = tokenizer.init_kwargs['mask_token_sent']
word_mask = tokenizer.init_kwargs['mask_token']
print(f'sent_mask {sent_mask}   word_mask {word_mask}')

sent_percentage = 0.15
word_percentage = 0.15

import random
import nltk
from nltk import tokenize
nltk.download('punkt')
def get_data(articles):
    encoder_input, decoder_target = [], []

    for article in articles:
        article_sents = tokenize.sent_tokenize(article)

        drop_ids = sorted(random.sample(range(len(article_sents)), int(sent_percentage * len(article_sents))))

        predict_sent = [article_sents[i] for i in drop_ids]
        for i in drop_ids:
            article_sents[i] = sent_mask

        encoder_input.append(' '.join(article_sents))
        decoder_target.append(' '.join(predict_sent))

    return encoder_input, decoder_target




def train_dataset(model, dataloader, scaler, criterion, optimizer, scheduler, run, DEVICE, _type='train', _break=1024):
    model.train()

    train_runnig_k = 25
    train_runnig_losses, train_runnig_nums = [], []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=_type, leave=False):
        optimizer.zero_grad()

        input_data, output_data = get_data(batch['article'])
        input_tokens = tokenizer(input_data, truncation=True, max_length=MAX_INPUT, padding='longest', return_tensors="pt")
        target_tokens = tokenizer(output_data, truncation=True, max_length=MAX_OUTPUT, padding='longest', return_tensors="pt")

        input_tokens = {k: v.to(DEVICE) for k, v in input_tokens.items()}
        target_tokens = {k: v.to(DEVICE) for k, v in target_tokens.items()}
        node_features = [i.to(DEVICE) for i in batch['node_features']]
        topology = [i.to(DEVICE) for i in batch['topology']]

        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(target_tokens['input_ids'])

        with torch.cuda.amp.autocast():
            try:
                outputs = model(**input_tokens, decoder_input_ids=decoder_input_ids, input_nodes_embeddings=node_features, input_edges=topology)
                loss = criterion(outputs.logits, target_tokens['input_ids'], target_tokens['attention_mask'])
            except RuntimeError:
                for k, v in input_tokens.items(): del v
                for k, v in target_tokens.items(): del v
                for v in node_features: del v
                for v in topology: del v
                del decoder_input_ids
                continue

        if torch.isnan(loss).any():
            print('loss NAN')
            exit(1)

        for k, v in input_tokens.items(): del v
        for k, v in target_tokens.items(): del v
        for v in node_features: del v
        for v in topology: del v
        del decoder_input_ids

        train_runnig_losses.append(loss.item())
        train_runnig_losses = train_runnig_losses[-train_runnig_k:]

        if scheduler:
            run[f"{_type}/lr"].log(scheduler.get_last_lr())
        run[f"{_type}/loss"].log(loss.item())
        run[f"{_type}/running-loss"].log(sum(train_runnig_losses) / len(train_runnig_losses))

        if i % (_break//2) == 0:
            out_summary = tokenizer.batch_decode(torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1))
            run[f"{_type}/out"].log(out_summary)



        scaler.scale(loss).backward()  #
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)  #
        # loss.backward()
        # optimizer.step()

        if scheduler:
            scheduler.step()
        scaler.update()

        del outputs
        del loss

        if i % 8 == 0:
            torch.cuda.empty_cache()

        if _break is not None and i > _break:
            break



@torch.no_grad()
def test_dataset(model, dataloader, run, DEVICE, _type='val', _break=64):
    model.eval()

    val_runnig_loss, val_runnig_num = 0, 0
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=_type, leave=False):
        input_data, output_data = get_data(batch['article'])
        input_tokens = tokenizer(input_data, truncation=True, max_length=MAX_INPUT, padding='longest', return_tensors="pt")
        target_tokens = tokenizer(output_data, truncation=True, max_length=MAX_OUTPUT, padding='longest', return_tensors="pt")

        input_tokens = {k: v.to(DEVICE) for k, v in input_tokens.items()}
        target_tokens = {k: v.to(DEVICE) for k, v in target_tokens.items()}
        node_features = [i.to(DEVICE) for i in batch['node_features']]
        topology = [i.to(DEVICE) for i in batch['topology']]

        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(target_tokens['input_ids'])

        outputs = model(**input_tokens, decoder_input_ids=decoder_input_ids, input_nodes_embeddings=node_features, input_edges=topology)
        loss = criterion(outputs.logits, target_tokens['input_ids'], target_tokens['attention_mask'])

        val_runnig_loss += loss.item()
        val_runnig_num += 1

        out_summary = tokenizer.batch_decode(torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1))
        if i == 0:
            run[f"{_type}/out"].log(out_summary)

        del decoder_input_ids
        del outputs
        del loss
        for k, v in input_tokens.items(): del v
        for k, v in target_tokens.items(): del v
        for v in node_features: del v
        for v in topology: del v

        if i % 8 == 0:
            torch.cuda.empty_cache()

        if _break is not None and i > _break:
            break

    run[f"{_type}/loss"].log(val_runnig_loss / val_runnig_num)

    return {'loss': val_runnig_loss / val_runnig_num}





test_dataset(model, dataloader_val, run, DEVICE, _type='val')
torch.cuda.empty_cache()

for e in tqdm(range(start_epoch, EPOCHS), desc="epoch", leave=False):
    train_dataset(model, dataloader_train, scaler, criterion, optimizer, scheduler, run, DEVICE)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    metrics = test_dataset(model, dataloader_val, run, DEVICE, _type='val')
    torch.cuda.empty_cache()
    checkpoint(model, optimizer, e, metrics=metrics)

