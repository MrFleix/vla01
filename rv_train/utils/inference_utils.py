import os

import torch

from rv_train.configs import get_cfg_defaults
from rv_train.train import get_cfg, get_model, load_model
from rv_train.utils.train_utils import short_name


def get_pretrained_model(
    model_path: str,
    device=0,
    torch_compile: bool = False,
):
    """
    Baut exakt das gleiche QwenActor-Modell wie im Training
    und lädt Checkpoint + Dataset-Stats korrekt.
    """

    device = f"cuda:{device}" if isinstance(device, int) else device

    # -------------------------------------------------
    # 1️⃣ Config laden
    # -------------------------------------------------
    model_folder = os.path.dirname(model_path)
    cfg_path = os.path.join(model_folder, "config.yaml")
    cfg = get_cfg(cfg_path, cfg_opts="")

    # -------------------------------------------------
    # 2️⃣ Modell bauen (Inference-Modus)
    # -------------------------------------------------
    model = get_model(
        cfg,
        calculate_dataset_stats=False,
        for_training=False,
    ).to(device)

    # -------------------------------------------------
    # 3️⃣ Checkpoint + Dataset Stats laden
    # -------------------------------------------------
    model, _ = load_model(
        model=model,
        model_path=model_path,
        cfg=cfg,
    )

    # -------------------------------------------------
    # 4️⃣ Optional torch.compile (NUR inneres Modell)
    # -------------------------------------------------
    if torch_compile:
        print("Compiling model.model with torch.compile...")
        model.model = torch.compile(model.model)

        if hasattr(model.model, "generate"):
            model.model.generate = torch.compile(model.model.generate)

    print("✅ Pretrained model ready for eval & inference")

    return model, cfg
