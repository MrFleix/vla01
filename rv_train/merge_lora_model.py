import argparse
import os
import shutil

from unsloth import FastVisionModel

from rv_train.train import get_cfg

"""
LoRA merge script that saves the merged full model and moves the LoRA adapter checkpoint.

Example commands:

# Standard merge (Full Model + LoRA Adapter, 16-bit merge)
python -m rv_train.merge_model_lora \
  --model_path ./runs/vla0/checkpoint-10

# Merge without actually merging the model, only creates folders and moves the LoRA adapter
python -m rv_train.merge_model_lora \
  --model_path ./runs/vla0/checkpoint-10 \
  --no_merge
"""


def merge_and_save_full_model(lora_checkpoint_dir: str, no_merge: bool = False):
    # Load config
    log_dir = os.path.dirname(lora_checkpoint_dir)
    cfg_path = os.path.join(log_dir, "config.yaml")
    stat_path = os.path.join(log_dir, "dataset_stats.pkl")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    cfg = get_cfg(cfg_path, cfg_opts="")

    # Extract values from config and path
    base_name = cfg.MODEL.UNSLOTH.qwen_model_id.split("/")[-1]
    peft_r = cfg.MODEL.UNSLOTH.peft_r
    batch_size = cfg.TRAIN.per_device_train_batch_size
    grad_accum = cfg.TRAIN.gradient_accumulation_steps
    horizon = cfg.MODEL.UNSLOTH.horizon

    folder_name = os.path.basename(lora_checkpoint_dir)
    if "checkpoint-" in folder_name:
        train_steps = folder_name.split("checkpoint-")[-1]
    else:
        train_steps = "0"

    # Determine save path and name for full model
    base_save_dir = os.path.join("runs/full_model", base_name)
    os.makedirs(base_save_dir, exist_ok=True)

    versioned_folder_template = (
        f"train_steps{train_steps}_lora{peft_r}_"
        f"Batch{batch_size}_GradAccu{grad_accum}_Horizon{horizon}"
    )

    final_save_path = os.path.join(base_save_dir, versioned_folder_template)

    # Versioning for full model
    version = 0
    while os.path.exists(final_save_path + f"_v{version}"):
        version += 1
    final_save_path += f"_v{version}"

    # Merge the model unless --no_merge is set
    if not no_merge:
        print(f"Loading full model from LoRA checkpoint: {lora_checkpoint_dir}")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=lora_checkpoint_dir, load_in_4bit=False, load_in_16bit=True
        )
        print(f"Saving merged model to {final_save_path}")
        model.save_pretrained_merged(final_save_path, tokenizer)
        print(f"Full model saved to {final_save_path}")
    else:
        print(
            f"No merge requested. Skipping model merge. Folder for version: {final_save_path}"
        )

    # Move LoRA checkpoint under full_model, with _ADAPTER and same version
    adapter_folder_name = versioned_folder_template + "_ADAPTER"
    adapter_final_path = os.path.join(
        base_save_dir, adapter_folder_name + f"_v{version}"
    )

    shutil.copytree(lora_checkpoint_dir, adapter_final_path)
    print(f"LoRA adapter checkpoint saved to {adapter_final_path}")

    # Copy config file under full_model
    config_final_path = os.path.join(base_save_dir, base_name + f"_config.yaml")
    shutil.copy(cfg_path, config_final_path)
    print(f"Config file copied to {config_final_path}")

    dataset_stats_path = os.path.join(base_save_dir, f"dataset_stats.pkl")
    shutil.copy(stat_path, dataset_stats_path)
    print(f"Config file copied to {dataset_stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the LoRA checkpoint directory (e.g., ./runs/vla0/checkpoint-10)",
    )
    parser.add_argument(
        "--no_merge",
        action="store_true",
        help="If set, skip merging the model; only create folders and move LoRA adapter",
    )
    args = parser.parse_args()
    merge_and_save_full_model(args.model_path, no_merge=args.no_merge)
