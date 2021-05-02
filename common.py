import torch
import torch.nn as nn


_criterion = nn.CrossEntropyLoss(reduction='none')

loss_fct = nn.CrossEntropyLoss()


def _get_criterion_loss(logits, target, mask=None):
    """
    logits: (batch, sent_len, tokens_total)
    target: (batch, sent_len)
    mask: (batch, sent_len)
    """

    masked_lm_loss = loss_fct(logits.view(-1, logits.size(-1)), target.view(-1))

    return masked_lm_loss


def get_criterion_loss(logits, target, mask=None):
    """
    logits: (batch, sent_len, tokens_total)
    target: (batch, sent_len)
    mask: (batch, sent_len)
    """
    _tokens_loss = _criterion(logits.permute(0, 2, 1), target)
    if mask is not None:
        _tokens_loss = _tokens_loss * mask
    _samples_loss = torch.sum(_tokens_loss.flatten(start_dim=1), axis=1)
    if mask is not None:
        _samples_loss = _samples_loss / torch.sum(mask, axis=1)
    else:
        _samples_loss = _samples_loss / _samples_loss.shape[0]

    loss = torch.mean(_samples_loss)

    if torch.isnan(loss).any():
        print('loss NAN', loss)
        print('  logits\n', logits)
        print('  target\n', target)
        print('  mask\n', mask)

    return loss

