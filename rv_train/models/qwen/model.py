import gc
import random
import warnings
from typing import List

import numpy as np
import PIL
import torch
from PIL import Image
from torch import nn
from transformers import (AutoConfig, LogitsProcessor, LogitsProcessorList,
                          Qwen2_5_VLProcessor)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
    Qwen2_5_VLForConditionalGeneration
from unsloth import FastVisionModel

import rv_train.constants as C
from rv_train.models.qwen.processor import get_imgs
from rv_train.utils.train_utils import ForkedPdb as debug  # noqa: F401


class NumberSpaceOnlyProcessor(LogitsProcessor):
    """Constrains generation to numbers (0-9), spaces, and EOS."""

    def __init__(self, processor):
        self.allowed_tokens = set()
        tokenizer = processor.tokenizer
        for i in range(10):
            self.allowed_tokens.add(
                tokenizer.encode(str(i), add_special_tokens=False)[0]
            )
        self.allowed_tokens.add(tokenizer.encode(" ", add_special_tokens=False)[0])
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)
        print("Jfjejfejfjejfejf")

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_tokens:
            mask[:, token_id] = 0
        return scores + mask


class QwenActor(nn.Module):
    def __init__(
        self,
        qwen_model_id,
        action_type,
        original_action_dim,
        horizon,
        history=1,
        use_lora=True,
        use_qlora=True,
        num_cam=2,
        lora_config="",
        lora_rank=8,
        rgb_input=False,
        rgb_img_size=(84, 84),
        add_vision_id=False,
        tiled_rgb_imgs=False,
        num_bins_actions=1000,
        use_flash_attention_2=True,
        system_message_version=1,
        action_mask_aug=0,
        action_mask_aug_per=0.1,
        attention_dropout=0.0,
        processor=None,
        # --- Unsloth / FastVisionModel ---
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        peft_r=16,
        peft_alpha=16,
        peft_dropout=0.0,
        peft_bias="none",
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        xformers=False,
        for_training=True,
        **kwargs,
    ):
        """
        :param qwen_model_id: str, the id of the qwen model to use
        :param action_type: str, the type of action to use, either ORIGINAL or EE
        :param original_action_dim: int, the dimension of the original action
        :param horizon: int, the horizon of the action
        :param history: int, the history of the action
        :param use_qlora: bool, whether to use qlora for parameter efficient fine-tuning
        :param num_cam: int, the number of cameras for rgb input
        :param lora_config: str, the lora configuration to use, either empty string or "default"
        :param lora_rank: int, the rank of the lora to use, only used if lora_config is "default"
        :param rgb_input: bool, whether to use rgb image input
        :param rgb_img_size: tuple, the size of the rgb image input (height, width)
        :param add_vision_id: bool, whether to add vision id to the input for qwen2.5
        :param tiled_rgb_imgs: bool, whether to tile the rgb images into a single image instead feeding them separately
        :param num_bins_actions: int, the number of bins in which each action dimension is discretized
        :param use_flash_attention_2: bool, whether to use flash attention 2 for faster training and inference
        :param attention_dropout: float, the dropout rate for the attention layer in the qwen model. Only tested when use_lora is False.
        """
        super(QwenActor, self).__init__()

        if history > 1 or num_cam > 1:
            assert (
                add_vision_id
            ), "add_vision_id must be True if history > 1 or num_cam > 1"
        self.processor = processor
        self.load_param_before_ddp = True
        self.qwen_model_id = qwen_model_id
        self.action_type = action_type
        self.original_action_dim = original_action_dim
        self.horizon = horizon
        self.history = history
        self.use_lora = use_lora
        self.use_qlora = use_qlora
        self.num_cam = num_cam
        self.lora_config = lora_config
        self.lora_rank = lora_rank
        self.rgb_input = rgb_input
        self.rgb_img_size = rgb_img_size
        self.add_vision_id = add_vision_id
        self.tiled_rgb_imgs = tiled_rgb_imgs
        self.num_bins_actions = num_bins_actions
        self.use_flash_attention_2 = use_flash_attention_2
        self.action_mask_aug_per = action_mask_aug_per
        self.attention_dropout = attention_dropout
        self.original_action_dim = original_action_dim
        self.xformers = xformers
        self.for_training = for_training
        # System Prompt

        if action_type == C.ORIGINAL:
            self.act_dim = original_action_dim
        elif action_type == C.EE:
            self.act_dim = 7
        else:
            assert False

        self.system_message = f"Analyze the input image and predict robot actions for the next {self.horizon} timesteps. Each action has {self.act_dim} dimensions. Output a single sequence of {self.horizon * self.act_dim} integers (0-{self.num_bins_actions} each), representing the {self.horizon} timesteps sequentially. Provide only space separated numbers. Nothing else."
        print("The System message is: ", self.system_message)

        # -----------------------------
        # Training vs Inference Laden
        # -----------------------------
        if for_training:
            # Training: Basismodell + ggf. LoRA aktivieren
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                qwen_model_id,
                load_in_4bit=load_in_4bit,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )
            # PEFT/LoRA nur beim Training
            self.model = FastVisionModel.get_peft_model(
                self.model,
                r=peft_r,
                lora_alpha=peft_alpha,
                lora_dropout=peft_dropout,
                bias=peft_bias,
                finetune_vision_layers=finetune_vision_layers,
                finetune_language_layers=finetune_language_layers,
                finetune_attention_modules=finetune_attention_modules,
                finetune_mlp_modules=finetune_mlp_modules,
            )
            FastVisionModel.for_training(self.model)
        else:
            # Inference: Full Merged Model enthält bereits LoRA
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                qwen_model_id,  # hier der merged model folder
                load_in_4bit=False,
            )
            FastVisionModel.for_inference(self.model)
            self.model.eval()  # sicherstellen, dass eval gesetzt ist
            self.logits_processor = NumberSpaceOnlyProcessor(self.tokenizer)
        # xFormers optional aktivieren
        if xformers:
            try:
                self.model.enable_xformers_memory_efficient_attention()
                print("Activate xformers")
            except Exception as e:
                print(f"Cannot find xformers: {e}")

        # Pixelgrenzen für Bildprozessor setzen
        self.min_pixel = self.max_pixel = self.rgb_img_size[0] * self.rgb_img_size[1]
        if self.rgb_input and self.tiled_rgb_imgs:
            self.min_pixel *= self.history * self.num_cam
            self.max_pixel *= self.history * self.num_cam

        if hasattr(self.tokenizer, "image_processor"):
            img_proc = self.tokenizer.image_processor

            print("Vorher:")
            print("  min_pixels =", getattr(img_proc, "min_pixels", None))
            print("  max_pixels =", getattr(img_proc, "max_pixels", None))

            img_proc.min_pixels = self.min_pixel
            img_proc.max_pixels = self.max_pixel

            print("Nachher:")
            print("  min_pixels =", img_proc.min_pixels)
            print("  max_pixels =", img_proc.max_pixels)
        else:
            print(
                "tokenizer hat kein image_processor – Pixelgrenzen können nicht gesetzt werden."
            )

    def set_dataset_stats(self, dataset_stats):
        """
        Set the dataset stats for the model
        :param dataset_stats: dict, the dataset stats
        """
        if dataset_stats == {}:
            warnings.warn(
                "Dataset stats is empty likely because the system does not have the data used to compute the stats. Ignore this is you are loading a pretrained model."
            )
            return

        self.original_dataset_stats = dataset_stats
        if self.action_type == C.ORIGINAL:
            self.dataset_stats = dataset_stats["out_ori_act"]
        else:
            raise NotImplementedError(f"Action type {self.action_type} not implemented")

    def get_min_max_act(self, instruction):
        """
        Get the min and max action for the instruction.
        :param instruction: str, the instruction for the current episode. This is needed for libero bounds 99% as the action space is different for different instructions.
        :return: torch.Tensor, the min and max action.
        """
        assert instruction is not None, "instruction is needed for libero bounds 99%"
        min_act = []
        max_act = []
        for _instruction in instruction:
            _suite = self.instruction_to_suite[_instruction]
            min_act.append(torch.tensor(self.dataset_stats[_suite]["min"]))
            max_act.append(torch.tensor(self.dataset_stats[_suite]["max"]))
        min_act = torch.stack(min_act, dim=0)
        max_act = torch.stack(max_act, dim=0)
        return min_act, max_act

    def get_text_action(self, actions, instruction=None):
        """
        Convert numerical actions into text actions for training.
        """
        # compute min/max if not cached or device mismatch
        if (
            (not hasattr(self, "_min_act"))
            or (not hasattr(self, "_max_act"))
            or (self._min_act.device != actions.device)
            or (self._max_act.device != actions.device)
        ):
            min_act = torch.tensor(self.dataset_stats["min"], device=actions.device)
            max_act = torch.tensor(self.dataset_stats["max"], device=actions.device)
            self._min_act = min_act
            self._max_act = max_act
        else:
            min_act = self._min_act
            max_act = self._max_act

        assert torch.all(min_act <= actions) and torch.all(
            actions <= max_act
        ), f"Action is out of range: {actions}"

        actions = (actions - min_act) / (max_act - min_act)
        actions *= self.num_bins_actions
        actions = torch.round(actions).long()
        actions = actions.reshape(actions.shape[0], -1)
        action_txt = [" ".join(map(str, x.tolist())) for x in actions]

        return action_txt

    def get_action_from_text_action(self, action_txt, instruction=None):
        """
        Convert text actions back into numerical actions.
        """
        bs = len(action_txt)
        min_act = torch.tensor(self.dataset_stats["min"])
        max_act = torch.tensor(self.dataset_stats["max"])

        try:
            action_txt = [x.strip() for x in action_txt]
            action = [
                [int(x) for x in _action_txt.split(" ")] for _action_txt in action_txt
            ]

            # pad/truncate to horizon
            action = torch.tensor(action, dtype=torch.float32).reshape(
                bs, -1, self.act_dim
            )
            if action.shape[1] < self.horizon:
                action = torch.cat(
                    [
                        action,
                        action[:, -1:].repeat(1, self.horizon - action.shape[1], 1),
                    ],
                    dim=1,
                )
            if action.shape[1] > self.horizon:
                action = action[:, : self.horizon]

            action = ((action / self.num_bins_actions) * (max_act - min_act)) + min_act

        except Exception as e:
            print(f"Error parsing action text: {e}")
            print(action_txt)
            action = ((min_act + max_act) / 2).repeat(bs, self.horizon, 1)

        return action

    @torch.no_grad()
    def forward(self, rgb, instruction=None, **kwargs):
        """
        LIBERO entry point.
        rgb: np.ndarray (H, W, 3)
        returns: action tensor [Horizon, act_dim]
        """
        if instruction is None:
            instruction = "Perform the task."

        return self.predict(rgb, instruction)

    @torch.no_grad()
    def predict(self, rgb, instruction: str, temperature: float = 1.5):
        """
        Predict actions from RGB images and an instruction using Qwen2.5-VL.
        rgb: np.ndarray or list of images (H, W, 3)
        instruction: str, instruction for the model
        """

        # 1️⃣ Bereite Bilder vor
        image = get_imgs(rgb, tile=self.tiled_rgb_imgs)

        if self.tiled_rgb_imgs:
            # Bei get_imgs als einzelnes Bild zurückkommt, in Liste packen
            if not isinstance(image, list):
                image = [image]

        # Debugging
        print("DEBUG: type(image) =", type(image))
        print(
            "DEBUG: len(image) =", len(image) if isinstance(image, list) else "single"
        )

        # 2️⃣ Konvertiere in Chat-Format mit <image> Token
        # add_vision_id=True erzeugt den <image> Token für das Bild
        input_text = self.tokenizer.apply_chat_template(
            self.convert_to_conversation2(image, instruction),
            add_generation_prompt=True,
            add_vision_id=True,
        )

        # 3️⃣ Tokenize Text (Text enthält nun <image> Token)
        inputs = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        # 4️⃣ Generiere Text (die Ausgabe sind die Zahlen als String)
        from transformers import LogitsProcessorList, TextStreamer

        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=temperature,
            logits_processor=LogitsProcessorList([self.logits_processor]),
        )

        # 5️⃣ Extrahiere generierten Text (nur die neuen Tokens)
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        action_text = self.tokenizer.decode(generated, skip_special_tokens=True)

        # 6️⃣ Konvertiere Text zu numerischen Actions
        return self._text_to_action(action_text)

    def convert_to_conversation2(self, instruction, images):
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": images},
                ],
            },
        ]
        return {"messages": conversation}

    def convert_to_conversation(self, sample):
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": sample["system"]}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample["instr"]},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
        return {"messages": conversation}

    def _text_to_action(self, text: str):
        """Convert action text to tensor."""
        if self.stats is None:
            raise ValueError("Stats not loaded")

        stats = self.stats["out_ori_act"]
        min_act = torch.tensor(stats["min"])
        max_act = torch.tensor(stats["max"])

        try:
            tokens = [int(x) for x in text.strip().split()]
            actions = torch.tensor(tokens, dtype=torch.float32)
            actions = actions.reshape(-1, self.action_dim)

            if actions.shape[0] < self.horizon:
                pad = actions[-1:].repeat(self.horizon - actions.shape[0], 1)
                actions = torch.cat([actions, pad], dim=0)
            actions = actions[: self.horizon]

            actions = (actions / self.num_bins) * (max_act - min_act) + min_act
        except Exception:
            actions = ((min_act + max_act) / 2).repeat(self.horizon, 1)

        return actions
