import random

import torch
from unsloth.trainer import UnslothVisionDataCollator


class MaskedVisionDataCollator(UnslothVisionDataCollator):
    def __init__(self, model, processor, action_mask_aug_per=0.2, **kwargs):
        super().__init__(model, processor, **kwargs)
        self.action_mask_aug_per = action_mask_aug_per

    def __call__(self, batch):
        # 1) Originalverarbeitung aufrufen
        # print(batch)
        batch = super().__call__(batch)

        new_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                new_batch[k] = v.clone()
            else:
                new_batch[k] = v

        input_ids = new_batch["input_ids"]
        labels = new_batch["labels"]

        instruction_mask = labels == self.ignore_index  # Instruction + PAD

        # probability for masking
        if random.random() < 0.1:
            _action_mask_aug_per = 0.0
        else:
            _action_mask_aug_per = random.uniform(0.0, self.action_mask_aug_per)

        # relevant tokens
        relevant_mask = (
            torch.logical_not(torch.isin(input_ids, self.padding_token_ids))
            & ~instruction_mask
        )

        rand_mask = torch.rand_like(input_ids, dtype=torch.float) < _action_mask_aug_per
        mask_tokens = rand_mask & relevant_mask

        # masking
        labels[mask_tokens] = self.ignore_index
        input_ids[mask_tokens] = 30  # '?'

        batch["input_ids"] = input_ids
        batch["labels"] = labels

        # 8) Optional: Debug
        ##self.print_batch_debug(batch)

        return batch

    def print_batch_debug(self, batch):
        torch.set_printoptions(profile="full")  # ganze Tensoren anzeigen
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attn_mask = batch["attention_mask"]

        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            print(f"--- Sample {i} ---")
            print("input_ids : ", input_ids[i].tolist())
            print(
                "labels    : ", ["-100" if l == -100 else l for l in labels[i].tolist()]
            )
            if attn_mask is not None:
                print("attn_mask : ", attn_mask[i].tolist())
            print("-" * 80)
