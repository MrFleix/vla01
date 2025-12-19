from yacs.config import CfgNode as CN

import rv_train.constants as C

_C = CN()

# ----------------------------------------------------------------------------
# MODEL SETTINGS: UNSLOTH / FASTVISION
# ----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.UNSLOTH = CN()
_C.MODEL.UNSLOTH.load_in_4bit = True
_C.MODEL.UNSLOTH.use_gradient_checkpointing = "unsloth"
_C.MODEL.UNSLOTH.peft_r = 16
_C.MODEL.UNSLOTH.peft_alpha = 16
_C.MODEL.UNSLOTH.peft_dropout = 0.0
_C.MODEL.UNSLOTH.peft_bias = "none"
_C.MODEL.UNSLOTH.finetune_vision_layers = True
_C.MODEL.UNSLOTH.finetune_language_layers = True
_C.MODEL.UNSLOTH.finetune_attention_modules = True
_C.MODEL.UNSLOTH.finetune_mlp_modules = True
_C.MODEL.UNSLOTH.action_type = C.ORIGINAL
_C.MODEL.UNSLOTH.original_action_dim = 7
_C.MODEL.UNSLOTH.horizon = 8
_C.MODEL.UNSLOTH.num_cam = 2
_C.MODEL.UNSLOTH.rgb_input = True
_C.MODEL.UNSLOTH.rgb_img_size = (224, 224)
_C.MODEL.UNSLOTH.add_vision_id = True
_C.MODEL.UNSLOTH.tiled_rgb_imgs = True
_C.MODEL.UNSLOTH.num_bins_actions = 1000
_C.MODEL.UNSLOTH.use_flash_attention_2 = False
_C.MODEL.UNSLOTH.action_mask_aug_per = 0.4
_C.MODEL.UNSLOTH.attention_dropout = 0.0
_C.MODEL.UNSLOTH.qwen_model_id = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"

# ----------------------------------------------------------------------------
# TRAINING / SFT Trainer
# ----------------------------------------------------------------------------
_C.TRAIN = CN()
# SFT parameters
_C.TRAIN.per_device_train_batch_size = 2
_C.TRAIN.gradient_accumulation_steps = 4
_C.TRAIN.warmup_steps = 5
_C.TRAIN.max_steps = -1
_C.TRAIN.learning_rate = 2e-4
_C.TRAIN.logging_steps = 1
_C.TRAIN.optim = "adamw_8bit"
_C.TRAIN.weight_decay = 0.001
_C.TRAIN.lr_scheduler_type = "linear"
_C.TRAIN.xformers = True
# Checkpoint settings
_C.TRAIN.save_steps = 500
_C.TRAIN.save_total_limit = 3
_C.TRAIN.seed = 3407
_C.TRAIN.output_dir = "./runs"
_C.TRAIN.report_to = "none"

# Dataset / tokenizer specifics
_C.TRAIN.remove_unused_columns = False
_C.TRAIN.dataset_text_field = "instr"
_C.TRAIN.dataset_kwargs = (("skip_prepare_dataset", True),)
_C.TRAIN.max_length = 2048

# Training details
_C.TRAIN.lr = 5e-6
_C.TRAIN.num_epochs = 24
_C.TRAIN.clip_grad_norm = 0.0

# ----------------------------------------------------------------------------
# EXPERIMENT SETTINGS
# ----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.EXP_ID = "vla0"
_C.EXP.DATASET = "roboverse"
_C.EXP.MODEL = "unsloth"
_C.EXP.OPTIMIZER = "adamw"
_C.EXP.LR_SCHED = "none"
_C.EXP.AMP = True
_C.EXP.LOSS = CN()  # optional / leer

# Extra experiment parameters
_C.EXP_EXTRA = CN()
_C.EXP_EXTRA.save_ckp = 2
_C.EXP_EXTRA.no_val = True
_C.EXP_EXTRA.no_test = True
_C.EXP_EXTRA.no_track = True
_C.EXP_EXTRA.val_eval_freq = 1
_C.EXP_EXTRA.test_eval_freq = 1
_C.EXP_EXTRA.save_last_ckpt = True

# ----------------------------------------------------------------------------
# DATALOADER SETTINGS
# ----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.num_workers = 8

# ROBOVERSE / METAWORLD Dataset
_C.DATALOADER.ROBOVERSE = CN()
_C.DATALOADER.ROBOVERSE.cfg_path = (
    "libs/RoboVerse/roboverse/configs/img_libero_aug.yaml"
)
_C.DATALOADER.ROBOVERSE.cfg_opts = (
    "IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:('3p1','3p2')"
)


# ----------------------------------------------------------------------------
# DEFAULT CONFIG FUNCTION
# ----------------------------------------------------------------------------
def get_cfg_defaults():
    """Return a clone of the default config"""
    return _C.clone()
