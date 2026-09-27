"""Image classification by fine-tuning pretrained CNNs on the GPU. One call
trains one architecture; the backend runs architectures in parallel on
separate GPUs.

The recipe is fixed rather than searched, because each search trial would be
a full fine-tuning run: AdamW, a one-epoch warm-up then cosine decay, label
smoothing, bf16 mixed precision, channels-last memory layout, and early
stopping on a validation split. Imbalanced classes get a class-weighted loss
and macro-F1 model selection, the same plan as tabular data.

Splits are stratified: 70% train, 15% validation (early stopping and
checkpoint choice), 15% test (reported once, never used for decisions).
"""
import io
import os
import tempfile
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.training.common import RANDOM_STATE, full_metrics, library_versions, python_value, score
from app.training.runtime import BUNDLE_FORMAT

# Display name -> timm model (ImageNet-pretrained weights, baked into the GPU image).
ARCHITECTURES = {
    "ConvNeXt-Tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "EfficientNet-B0": "efficientnet_b0.ra_in1k",
    "ResNet-50": "resnet50.a1_in1k",
}
CANDIDATES = dict(ARCHITECTURES)
PRETRAINED = True
MAX_EPOCHS = 20
PATIENCE = 4
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
MIN_IMAGES_PER_CLASS = 5


class _Images:
    def __init__(self, root, paths, labels, transform):
        self.root, self.paths, self.labels, self.transform = root, paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image, ImageOps

        with Image.open(os.path.join(self.root, self.paths[i])) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            return self.transform(img), self.labels[i]


def split(y: np.ndarray):
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=TEST_FRACTION, random_state=RANDOM_STATE, stratify=y)
    train, val = train_test_split(rest, test_size=VAL_FRACTION / (1 - TEST_FRACTION),
                                  random_state=RANDOM_STATE, stratify=y[rest])
    return train, val, test


def _predict(net, loader, device):
    import torch

    net.eval()
    probs, labels = [], []
    use_amp = device.startswith("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for x, y in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            probs.append(torch.softmax(net(x).float(), dim=1).cpu().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def run(data: bytes, spec: dict, device: str) -> dict:
    """data: zip with dataset.parquet (columns image, label) and the image
    files it references. spec: {"candidate", "plan": {"imbalance"}}"""
    import timm
    import torch
    from torch.utils.data import DataLoader

    started = time.monotonic()
    name = spec["candidate"]
    imbalance = (spec.get("plan") or {}).get("imbalance") or {}
    balanced = imbalance.get("strategy") == "balanced_class_weights"
    metric = imbalance.get("selection_metric") or "accuracy"
    torch.manual_seed(RANDOM_STATE)
    use_amp = device.startswith("cuda")

    with tempfile.TemporaryDirectory() as root:
        zipfile.ZipFile(io.BytesIO(data)).extractall(root)
        manifest = pd.read_parquet(os.path.join(root, "dataset.parquet"))
        classes, y = np.unique(manifest["label"].astype(str).to_numpy(), return_inverse=True)
        counts = np.bincount(y)
        if counts.min() < MIN_IMAGES_PER_CLASS:
            small = [str(classes[i]) for i in np.flatnonzero(counts < MIN_IMAGES_PER_CLASS)]
            raise ValueError(f"Classes {small} have fewer than {MIN_IMAGES_PER_CLASS} images.")
        train_idx, val_idx, test_idx = split(y)
        paths = manifest["image"].tolist()

        net = timm.create_model(CANDIDATES[name], pretrained=PRETRAINED, num_classes=len(classes))
        data_config = timm.data.resolve_data_config({}, model=net)
        train_tf = timm.data.create_transform(**data_config, is_training=True)
        eval_tf = timm.data.create_transform(**data_config, is_training=False)
        workers = min(8, os.cpu_count() or 1) if use_amp else 0

        def loader(idx, tf, shuffle):
            return DataLoader(_Images(root, [paths[i] for i in idx], y[idx].tolist(), tf), batch_size=BATCH_SIZE,
                              shuffle=shuffle, num_workers=workers, pin_memory=use_amp)

        train_dl, val_dl, test_dl = loader(train_idx, train_tf, True), loader(val_idx, eval_tf, False), \
            loader(test_idx, eval_tf, False)

        net = net.to(device).to(memory_format=torch.channels_last)
        weight = None
        if balanced:
            freq = np.bincount(y[train_idx], minlength=len(classes)).astype(np.float64)
            weight = torch.tensor(len(train_idx) / (len(classes) * freq), dtype=torch.float32, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=LABEL_SMOOTHING)
        opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = max(1, len(train_dl))
        total = MAX_EPOCHS * steps_per_epoch

        def lr_at(step):  # one warm-up epoch, then cosine decay to zero
            if step < steps_per_epoch:
                return (step + 1) / steps_per_epoch
            return 0.5 * (1 + np.cos(np.pi * (step - steps_per_epoch) / max(1, total - steps_per_epoch)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        best, best_state, stale, epochs = -np.inf, None, 0, 0
        history = []
        for epoch in range(MAX_EPOCHS):
            net.train()
            for x, target in train_dl:
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                target = torch.as_tensor(target, device=device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = loss_fn(net(x), target)
                loss.backward()
                opt.step()
                sched.step()
            probs, labels = _predict(net, val_dl, device)
            val_score = score(metric, labels, probs.argmax(axis=1))
            history.append({"epoch": epoch + 1, "val_score": round(val_score, 4)})
            epochs = epoch + 1
            if val_score > best + 1e-4:
                best, stale = val_score, 0
                best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

        net.load_state_dict(best_state)
        test_probs, test_labels = _predict(net, test_dl, device)

    test_pred = test_probs.argmax(axis=1)
    state = io.BytesIO()
    torch.save(best_state, state)
    buf = io.BytesIO()
    class_labels = [python_value(c) for c in classes]
    joblib.dump({
        "format": BUNDLE_FORMAT, "task": "image_classification", "problem_type": "classification",
        "model_name": name, "feature_cols": [], "class_labels": class_labels,
        "estimator": {"kind": "timm", "arch": CANDIDATES[name], "state_dict": state.getvalue(),
                      "data_config": dict(data_config)},
    }, buf)
    return {
        "task": "image_classification", "candidate": name, "selection_metric": metric,
        "val_score": round(float(best), 4), "epochs": epochs, "history": history,
        "test_metrics": full_metrics("classification", test_labels, test_pred, test_probs),
        "test_score": score(metric, test_labels, test_pred),
        "class_labels": class_labels, "split_sizes": {"train": len(train_idx), "val": len(val_idx),
                                                      "test": len(test_idx)},
        "imbalance_applied": {"class_weights": balanced, "selection_metric": metric},
        "y_test": test_labels.tolist(), "y_pred": test_pred.tolist(),
        "bundle_bytes": buf.getvalue(), "library_versions": library_versions(), "device": device,
        "seconds": round(time.monotonic() - started, 1),
    }
