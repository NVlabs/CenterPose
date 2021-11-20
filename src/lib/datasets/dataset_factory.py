from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def collate_fn_filtered(batch):
    # In some rare cases, the whole batch will be empty
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) > 0:
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return None
