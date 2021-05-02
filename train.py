import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"



from dataset import CnnDmDataset
from torch.utils.data import DataLoader


dataset_train = CnnDmDataset('train')
dataset_val = CnnDmDataset('validation')



# BATCH_SIZE = 6
BATCH_SIZE = 3
BATCH_SIZE = 2
# BATCH_SIZE = 4
BATCH_SIZE_TEST = 5
# BATCH_SIZE_TEST = 8


dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, collate_fn=CnnDmDataset.collate_fn)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=8, collate_fn=CnnDmDataset.collate_fn)




from datasets import load_metric

metric_rouge = load_metric('rouge')
metric_meteor = load_metric('meteor')



from models import get_model

tokenizer, model = get_model(graph=False)





import torch
import  torch.nn as nn
import  torch.nn.functional as F
from tqdm.auto import tqdm


from common import get_criterion_loss, _get_criterion_loss
import neptune.new as neptune

run = neptune.init(project='k4black/diploma-sum', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZDliYzEyZS1hOWVkLTQ1ZDQtOThlYS1jNDhhOTFjMGQ4ZjAifQ==')
run["sys/tags"].add(['pretrain'])

from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

LR = 1e-4
EPOCHS = 21
start_epoch = 0
DEVICE = 'cuda'
# DEVICE = 'cpu'


scaler = torch.cuda.amp.GradScaler()  # mixed precision

criterion = lambda x, y, m: get_criterion_loss(x, y, m)
optimizer = AdamW(model.parameters(), lr=LR)
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1024, num_training_steps=2*1024*EPOCHS)
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=1024, num_training_steps=2*1024*EPOCHS, num_cycles=4)


# CHECKPOINT = 'model-best'
CHECKPOINT = None
# CHECKPOINT = 'model-best'
if CHECKPOINT is not None:
    print(f'LOADING checkpoint <{CHECKPOINT}>...')
    checkpoint = torch.load(f'checkpoints/{CHECKPOINT}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss'] if 'loss' in checkpoint else None
    rouge1 = checkpoint['rouge1'] if 'rouge1' in checkpoint else None
    print(f'    saved loss: {loss}')
    print(f'    saved rouge1: {rouge1}')

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


# for param in model.parameters():
#     param.requires_grad = True
# for param in model.shared.parameters():
#     param.requires_grad = False

from torchinfo import summary
summary(model, depth=3)


run["parameters"] = {
    "learning_rate": LR,
    "criterion": "CrossEntropyLoss",
    "optimizer": type(optimizer).__name__,
    # "model": "pegasus+gat",
    "model": "pegasus",
    "encoders": "5",
    # "gat": "3",
    "decoders": "3",
}


def checkpoint(model, optimizer, epoch, metrics=None):
    if metrics is not None:
        filename = f'checkpoints/model-e_{epoch}-l_{metrics["loss"]:.4f}-r1_{metrics["r1"]*100:.2f}.pt'
    else:
        filename = f'checkpoints/model-e_{epoch}.pt'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': metrics["loss"] if metrics else None,
        'rouge1': metrics["r1"] if metrics else None,
    }, filename)


def train_dataset(model, dataloader, scaler, criterion, optimizer, scheduler, run, DEVICE, _type='train', _break=1024):
    model.train()

    train_runnig_k = 25
    train_runnig_losses, train_runnig_nums = [], []
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=_type, leave=False):
        optimizer.zero_grad()

        input_tokens = tokenizer(batch['article'], truncation=True, padding='longest', return_tensors="pt")
        target_tokens = tokenizer(batch['summary'], truncation=True, padding='longest', return_tensors="pt")

        input_tokens = {k: v.to(DEVICE) for k, v in input_tokens.items()}
        target_tokens = {k: v.to(DEVICE) for k, v in target_tokens.items()}
        node_features = [i.to(DEVICE) for i in batch['node_features']]
        topology = [i.to(DEVICE) for i in batch['topology']]

        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(target_tokens['input_ids'])

        with torch.cuda.amp.autocast():
            outputs = model(**input_tokens, decoder_input_ids=decoder_input_ids, input_nodes_embeddings=node_features, input_edges=topology)
            loss = criterion(outputs.logits, target_tokens['input_ids'], target_tokens['attention_mask'])

        for k, v in input_tokens.items(): del v
        for k, v in target_tokens.items(): del v
        for v in node_features: del v
        for v in topology: del v
        del decoder_input_ids

        train_runnig_losses.append(loss.item())
        train_runnig_losses = train_runnig_losses[-train_runnig_k:]

        run[f"{_type}/lr"].log(scheduler.get_last_lr())
        run[f"{_type}/loss"].log(loss.item())
        run[f"{_type}/running-loss"].log(sum(train_runnig_losses) / len(train_runnig_losses))

        if i % (_break//2) == 0:
            out_summary = tokenizer.batch_decode(torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1))
            run[f"{_type}/out"].log(out_summary)



        scaler.scale(loss).backward()  #
        scaler.step(optimizer)  #
        # loss.backward()
        # optimizer.step()

        scheduler.step()
        scaler.update()

        del outputs
        del loss

        if i % 8 == 0:
            torch.cuda.empty_cache()

        if i > _break:
            break



@torch.no_grad()
def test_dataset(model, dataloader, run, DEVICE, _type='val', _break=64+32):
    model.eval()

    val_runnig_loss, val_runnig_num = 0, 0
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=_type, leave=False):
        input_tokens = tokenizer(batch['article'], truncation=True, padding='longest', return_tensors="pt")
        target_tokens = tokenizer(batch['summary'], truncation=True, padding='longest', return_tensors="pt")

        input_tokens = {k: v.to(DEVICE) for k, v in input_tokens.items()}
        target_tokens = {k: v.to(DEVICE) for k, v in target_tokens.items()}
        node_features = [i.to(DEVICE) for i in batch['node_features']]
        topology = [i.to(DEVICE) for i in batch['topology']]

        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(target_tokens['input_ids'])

        outputs = model(**input_tokens, decoder_input_ids=decoder_input_ids, input_nodes_embeddings=node_features, input_edges=topology)
        loss = criterion(outputs.logits, target_tokens['input_ids'], target_tokens['attention_mask'])

        val_runnig_loss += loss.item()
        val_runnig_num += 1

        if i == 0:
            out_summary = tokenizer.batch_decode(torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1))
            run[f"{_type}/out"].log(out_summary)

        del decoder_input_ids
        del outputs
        del loss
        # out_summary = tokenizer.batch_decode(torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1))
        # try:
        #     predict = model.module.predict(**input_tokens, input_nodes_embeddings=node_features, input_edges=topology)
        # except AttributeError:

        predict = model.predict(**input_tokens, input_nodes_embeddings=node_features, input_edges=topology)
        out_summary = tokenizer.batch_decode(predict.to('cpu'), skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # print('out_summary', out_summary)
        # del predict
        # out_summary = tokenizer.batch_decode(model.predict(**input_tokens).to('cpu'))
        metric_rouge.add_batch(predictions=out_summary, references=batch['summary'])
        metric_meteor.add_batch(predictions=out_summary, references=batch['summary'])

        if i == 0:
            run[f'{_type}/text'].log(out_summary[0])

        for k, v in input_tokens.items(): del v
        for k, v in target_tokens.items(): del v
        for v in node_features: del v
        for v in topology: del v
        del predict

        if i % 8 == 0:
            torch.cuda.empty_cache()

        if i > _break:
            break

    run[f"{_type}/loss"].log(val_runnig_loss / val_runnig_num)

    rouge = metric_rouge.compute()
    run[f"{_type}/rouge1"].log(rouge['rouge1'].mid.fmeasure)
    run[f"{_type}/rouge2"].log(rouge['rouge2'].mid.fmeasure)
    run[f"{_type}/rougeL"].log(rouge['rougeL'].mid.fmeasure)
    run[f"{_type}/rougeLsum"].log(rouge['rougeLsum'].mid.fmeasure)

    meteor = metric_meteor.compute()['meteor']
    run[f"{_type}/meteor"].log(meteor)

    return {'loss': val_runnig_loss / val_runnig_num, 'r1': rouge['rouge1'].mid.fmeasure}





test_dataset(model, dataloader_val, run, DEVICE, _type='val')
torch.cuda.empty_cache()

for e in tqdm(range(start_epoch, EPOCHS), desc="epoch", leave=False):
    train_dataset(model, dataloader_train, scaler, criterion, optimizer, scheduler, run, DEVICE)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    metrics = test_dataset(model, dataloader_val, run, DEVICE, _type='val')
    torch.cuda.empty_cache()
    checkpoint(model, optimizer, e, metrics=metrics)

