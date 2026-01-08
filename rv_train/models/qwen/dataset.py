import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from rv_train.models.qwen.processor import get_imgs, get_qwen_inputs
import os
import torch.distributed as dist



class QwenSFTDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        model,
        instruction="",
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

class QwenCachedDataset(Dataset):
    def __init__(self, cache_file, preprocess_fn=None, chunks_in_memory=2):
        """
        cache_file: Path to the starting cache part, e.g., "cache_checkpoints/cache_part_00000.pkl"
        preprocess_fn: Function to create cache if it does not exist
        chunks_in_memory: Number of chunks to keep in RAM at the same time
        """
        self.cache_file = cache_file
        self.cache_dir = os.path.dirname(cache_file)
        self.chunks_in_memory = chunks_in_memory

        # Distributed info
        self.rank = 0
        self.world_size = 1
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        # If starting chunk missing → run preprocessing
        if self.rank == 0 and not os.path.exists(cache_file):
            if preprocess_fn is None:
                raise RuntimeError(f"Start cache file {cache_file} not found and no preprocess_fn provided")
            print(f"[Rank {self.rank}] Start cache not found, running preprocessing...")
            os.makedirs(self.cache_dir, exist_ok=True)
            preprocess_fn()
            print(f"[Rank {self.rank}] Preprocessing done.")

        # Wait for all ranks
        if self.world_size > 1:
            dist.barrier()

        # Find all chunks
        start_num = int(os.path.basename(cache_file).split("_")[-1].split(".")[0])
        all_parts = sorted([
            f for f in os.listdir(self.cache_dir)
            if f.startswith("cache_part_") and f.endswith(".pkl")
        ])
        self.cache_parts = [
            os.path.join(self.cache_dir, f)
            for f in all_parts
            if int(f.split("_")[-1].split(".")[0]) >= start_num
        ]
        if not self.cache_parts:
            raise RuntimeError(f"No cache parts found in {self.cache_dir} starting from {cache_file}")

        print(f"[Rank {self.rank}] Found {len(self.cache_parts)} cache parts starting from {cache_file}")

        # Initialization
        self.loaded_chunks = {}  # {idx: [samples]}
        self.chunk_order = list(range(len(self.cache_parts)))  # cyclic order
        self.chunk_pointer = 0
        self.current_epoch = 0
        self.total_samples = sum([self._chunk_len(i) for i in range(len(self.cache_parts))])
        self.indices_in_chunk = {}  # track position inside each chunk

        # Load the first N chunks into RAM
        self._load_initial_chunks()

    def _chunk_len(self, chunk_idx):
        """Helper to get chunk length without keeping it in memory"""
        path = self.cache_parts[chunk_idx]
        with open(path, "rb") as f:
            part = pickle.load(f)
        return len(part)

    def _load_chunk(self, chunk_idx):
        path = self.cache_parts[chunk_idx]
        with open(path, "rb") as f:
            part = pickle.load(f)
            data = [x[1] for x in part]  # only samples
        self.loaded_chunks[chunk_idx] = data
        self.indices_in_chunk[chunk_idx] = 0
        print(f"[Rank {self.rank}] Loaded: {path} ({len(data)} samples)")

    def _unload_chunk(self, chunk_idx):
        if chunk_idx in self.loaded_chunks:
            del self.loaded_chunks[chunk_idx]
            del self.indices_in_chunk[chunk_idx]

    def _load_initial_chunks(self):
        # Load first N chunks into RAM
        for _ in range(min(self.chunks_in_memory, len(self.cache_parts))):
            idx = self.chunk_order[self.chunk_pointer]
            self._load_chunk(idx)
            self.chunk_pointer = (self.chunk_pointer + 1) % len(self.cache_parts)

    def _next_chunk_to_load(self):
        return self.chunk_order[self.chunk_pointer]

    def __len__(self):
        # Length = sum of all samples in all chunks
        return self.total_samples

    def __getitem__(self, idx):
        """
        Returns samples from loaded chunks.
        Loads new chunks when needed and unloads old ones to save RAM.
        """
        # Loop through loaded chunks
        for chunk_idx in list(self.loaded_chunks.keys()):
            pos = self.indices_in_chunk[chunk_idx]
            if pos < len(self.loaded_chunks[chunk_idx]):
                sample = self.loaded_chunks[chunk_idx][pos]
                self.indices_in_chunk[chunk_idx] += 1
                return sample
            else:
                # Chunk exhausted → unload
                self._unload_chunk(chunk_idx)

        # Load next chunk if needed
        next_chunk_idx = self._next_chunk_to_load()
        self._load_chunk(next_chunk_idx)
        self.chunk_pointer = (self.chunk_pointer + 1) % len(self.cache_parts)

        # If we looped back to start → epoch finished
        if self.chunk_pointer == 0:
            self.current_epoch += 1
            print(f"[Rank {self.rank}] Epoch {self.current_epoch} completed.")

        # Recursively fetch the sample
        return self.__getitem__(idx)











