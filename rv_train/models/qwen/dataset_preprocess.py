import time
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from rv_train.models.qwen.processor import get_imgs, get_qwen_inputs
import torch.distributed as dist
import os
import pickle
from rv_train.models.qwen.processor import get_imgs, get_qwen_inputs



def preprocess_qwen_dataset(
    base_dataset,
    model,
    max_workers=4,
    chunk_size=8,
    progress_step=250,
    checkpoint_step=20000,
    checkpoint_dir="./cache_checkpoints",
):
    """
    RAM-safe preprocessing:
    - no global growing cache
    - streaming to disk
    - constant memory usage
    - saves checkpoints as .pt for faster loading
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    total = len(base_dataset)

    start_time = time.time()
    completed = 0
    part_id = 0
    buffer = []

    print(f"Preprocessing {total} samples ...")

    def process_one(idx):
        item = base_dataset[idx]
        instr = item.get("instr", None)

        imgs = get_imgs(item["rgb"])

        action_txt_list = model.get_text_action(
            item.get("out_ori_act", []),
            instruction=instr,
        )
        action_txt = " ".join(action_txt_list)

        sample = {
            "image": imgs,
            "text": action_txt,
            "instr": instr,
            "system": model.system_message,
        }

        converted = model.convert_to_conversation(sample)
        return idx, converted

    indices = list(range(total))
    chunks = [indices[i:i + chunk_size] for i in range(0, total, chunk_size)]

    for chunk in chunks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, idx) for idx in chunk]

            for future in as_completed(futures):
                idx, sample = future.result()
                buffer.append((idx, sample))
                completed += 1

                if completed % progress_step == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = (total - completed) / rate if rate > 0 else 0.0
                    print(
                        f"{completed}/{total} "
                        f"({completed / total * 100:.2f}%) "
                        f"elapsed {elapsed:.1f}s "
                        f"ETA {eta:.1f}s "
                        f"{rate:.1f} it/s"
                    )

                if len(buffer) >= checkpoint_step:
                    buffer.sort(key=lambda x: x[0])
                    save_path = os.path.join(
                        checkpoint_dir,
                        f"cache_part_{part_id:05d}.pt",  # <- .pt
                    )
                    # Speichern als PyTorch-Datei
                    torch.save(buffer, save_path)
                    buffer.clear()
                    torch.cuda.empty_cache()
                    print(f"Checkpoint saved: {save_path}")
                    part_id += 1

    if buffer:
        buffer.sort(key=lambda x: x[0])
        save_path = os.path.join(
            checkpoint_dir,
            f"cache_part_{part_id:05d}.pt",
        )
        torch.save(buffer, save_path)
        buffer.clear()
        torch.cuda.empty_cache()
        print(f"Final checkpoint saved: {save_path}")

    print("Preprocessing finished.")

    # Merge zu einem finalen .pt, falls Rank 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        merge_cache_checkpoints(
            checkpoint_dir=checkpoint_dir,
            output_path="cache/qwen_precomputed.pt",
        )

    if dist.is_initialized():
        dist.barrier()


def merge_cache_checkpoints(checkpoint_dir, output_path):
    print("Merging cache checkpoints...")

    cache = []

    files = sorted(
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("cache_part_") and f.endswith(".pt")  # <- .pt
    )

    if not files:
        raise RuntimeError("No cache checkpoints found!")

    for fname in files:
        path = os.path.join(checkpoint_dir, fname)
        part = torch.load(path)
        cache.extend(part)

    # Sort by original index
    cache.sort(key=lambda x: x[0])

    # Strip the index
    cache = [sample for _, sample in cache]

    torch.save(cache, output_path)
    print(f"Final merged checkpoint saved: {output_path}")
    print(f"Total samples: {len(cache)}")
