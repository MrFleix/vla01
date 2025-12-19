import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from rv_train.models.qwen.processor import get_imgs, get_qwen_inputs


class QwenSFTDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        model,
        instruction="Write the LaTeX representation for this image.",
    ):
        self.dataset = base_dataset
        self.model = model
        self.instruction = instruction
        self._stats = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        instr = item.get("instr", self.instruction)
        rgb = item["rgb"]

        # rgb -> PIL Images
        imgs = get_imgs(
            rgb,
        )

        # Action-Text vom Model
        action_txt_list = self.model.get_text_action(
            item.get("out_ori_act", []), instruction=instr
        )
        action_txt = " ".join(action_txt_list)

        # Conversation-Format
        sample = {
            "image": imgs,  # falls mehrere Bilder
            "text": action_txt,
            "instr": instr,
            "system": self.model.system_message,
        }

        converted_dataset = self.model.convert_to_conversation(sample)
        # print(converted_dataset)
        return converted_dataset

    @property
    def stats(self):
        if self._stats is not None:
            return self._stats
        all_actions = []
        for item in self.dataset:
            if "out_ori_act" in item:
                all_actions.append(np.array(item["out_ori_act"], dtype=np.float32))
        if len(all_actions) == 0:
            self._stats = {"min": None, "max": None}
        else:
            all_actions = np.stack(all_actions)
            self._stats = {
                "min": all_actions.min(axis=0).tolist(),
                "max": all_actions.max(axis=0).tolist(),
            }
        return self._stats
