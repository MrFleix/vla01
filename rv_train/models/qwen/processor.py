# processor.py
from typing import List

import numpy as np
import torch
from PIL import Image


def get_imgs2(rgb):
    """
    rgb: np.ndarray, shape (num_cam, H, W, C) oder (batch, num_cam, H, W, C)
    return: list of PIL.Image.Image
    """
    # Falls Batch-Dim vorhanden: (1, num_cam, H, W, C) -> (num_cam, H, W, C)
    if rgb.ndim == 5:
        rgb = rgb[0]

    imgs = []
    for cam_img in rgb:  # cam_img: (H, W, C)
        # Falls Werte float 0-1, in uint8 umwandeln
        if cam_img.dtype != np.uint8:
            cam_img = (cam_img * 255).astype(np.uint8)
        # In PIL Image umwandeln
        # cam_img = torch.tensor(cam_img, dtype=torch.float32).permute(2,0,1)
        pil_img = Image.fromarray(cam_img)
        print("PIL Image size:", pil_img.size)  # (W, H)
        imgs.append(pil_img)

    return imgs


import numpy as np
import torch
from PIL import Image


def get_imgs_hm(rgb, tile=True):
    """
    rgb: torch.Tensor oder np.ndarray, shape (batch, history, num_cam, H, W, C)
    tile: bool, ob alle Kamerabilder zu einem Bild zusammengefügt werden sollen
    return: list of PIL.Image.Image pro Batch, oder ein einzelnes PIL.Image.Image, wenn tile=True
    """
    # Wenn Batch=1 & History=1: auf (num_cam, H, W, C) reduzieren
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()  # Tensor -> NumPy
    if rgb.ndim == 6:  # batch, history, num_cam, H, W, C
        rgb = rgb[0, 0]  # -> shape (num_cam, H, W, C)

    imgs = []
    for cam_img in rgb:  # cam_img: (H, W, C)
        # float -> uint8 falls nötig
        if cam_img.dtype != np.uint8:
            cam_img = (cam_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(cam_img)
        imgs.append(pil_img)

    # Optional: alle Kamerabilder zu einem Bild zusammenfügen
    if tile and len(imgs) > 1:
        widths, heights = zip(*(im.size for im in imgs))
        total_width = sum(widths)
        max_height = max(heights)
        dst = Image.new("RGB", (total_width, max_height))
        current_x = 0
        for im in imgs:
            dst.paste(im, (current_x, 0))
            current_x += im.width
        return dst

    return imgs  # Liste von PIL-Bildern


def get_imgs(rgb, tile=True):
    """
    rgb: np.ndarray, shape (num_cam, H, W, C) oder (batch, num_cam, H, W, C)
    tile: bool, ob alle Kamerabilder zu einem Bild zusammengefügt werden sollen
    return: list of PIL.Image.Image oder ein einzelnes PIL.Image.Image wenn tile=True
    """

    # print("DEBUG: type(rgb) =", type(rgb))
    # print("DEBUG: rgb.shape =", getattr(rgb, 'shape', None))
    # print("DEBUG: rgb.dtype =", getattr(rgb, 'dtype', None))
    # Falls Batch-Dim vorhanden: (1, num_cam, H, W, C) -> (num_cam, H, W, C)
    if rgb.ndim == 5:
        rgb = rgb[0]

    imgs = []
    for cam_img in rgb:  # cam_img: (H, W, C)
        if cam_img.dtype != np.uint8:
            cam_img = (cam_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(cam_img)
        imgs.append(pil_img)

    if tile and len(imgs) > 1:
        # Breite und Höhe berechnen
        widths, heights = zip(*(im.size for im in imgs))
        total_width = sum(widths)
        max_height = max(heights)

        # Neues Bild erstellen
        dst = Image.new("RGB", (total_width, max_height))
        current_x = 0
        for im in imgs:
            dst.paste(im, (current_x, 0))
            current_x += im.width
        return dst

    return imgs


def get_imgs2(rgb):
    """
    rgb: np.ndarray, shape (num_cam, H, W, C) oder (batch, num_cam, H, W, C)
    return: list of torch.Tensor, shape (3, H, W)
    """
    # Falls Batch-Dim vorhanden: (1, num_cam, H, W, C) -> (num_cam, H, W, C)
    if rgb.ndim == 5:
        rgb = rgb[0]

    imgs = []
    for cam_img in rgb:  # cam_img: (H, W, C)
        # Falls Werte float 0-1, in 0-255 umwandeln
        if cam_img.dtype != np.uint8:
            cam_img = (cam_img * 255).astype(np.uint8)
        # In Tensor konvertieren und permute von HWC -> CHW
        tensor_img = torch.tensor(cam_img, dtype=torch.float32).permute(
            2, 0, 1
        )  # C,H,W
        imgs.append(tensor_img)

    return imgs


def tile_images(images):
    """
    Tile a images into a single image
    :param images: Tensor of shape (bs, H, W, 3) or list of tensors of shape (H, W, 3)
    :return: Tensor of shape (bs, H, W, 3)
    """
    for img in images:
        assert len(img.shape) == 3, f"img.shape: {img.shape}"
        assert img.shape[2] == 3, f"img.shape: {img.shape}"

    widths, heights = zip(*(im.shape[:-1] for im in images))
    total_width = sum(widths)
    max_height = max(heights)
    dst = torch.zeros((max_height, total_width, 3), device=images[0].device)
    current_x = 0
    for i, img in enumerate(images):
        dst[: img.shape[0], current_x : current_x + img.shape[1], :] = img
        current_x += img.shape[1]
    return dst


def get_qwen_inputs(model, instr: List[str], imgs: List[List[Image.Image]]):
    """
    imgs: List[List[PIL.Image]] -> flache Liste pro Batch
    """
    # flach packen für den Tokenizer
    flat_imgs = [img for batch in imgs for img in batch]
    print(
        "Image Type: ", type(flat_imgs)
    )  # <class 'PIL.PngImagePlugin.PngImageFile'> oder <class 'PIL.Image.Image'>
    print("Image Size: ", flat_imgs.size)
    model_inputs = model.tokenizer(
        text=instr,
        images=flat_imgs,
        return_tensors="pt",
        padding=True,
        images_kwargs={"input_data_format": "channels_last"},  # <-- hier
    )

    # auf Modellgerät verschieben
    for k in model_inputs:
        model_inputs[k] = model_inputs[k].to(next(model.parameters()).device)

    return model_inputs
