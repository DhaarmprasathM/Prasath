import os
import sys
import types

# --------------------------------------------------
# 0. ROBUST FILE PATCHER & RUNTIME STUBS FOR VERSION MISMATCHES
# --------------------------------------------------
import torch

# Mock missing torch._C._dynamo module for newer torchvision on older PyTorch
try:
    if not hasattr(torch, '_C'):
        torch._C = types.ModuleType('torch._C')
        sys.modules['torch._C'] = torch._C

    if not hasattr(torch._C, '_dynamo'):
        dynamo_mod = types.ModuleType('torch._C._dynamo')
        dynamo_mod.__path__ = []
        sys.modules['torch._C._dynamo'] = dynamo_mod
        torch._C._dynamo = dynamo_mod

    if not hasattr(torch._C._dynamo, 'guards'):
        guards_mod = types.ModuleType('torch._C._dynamo.guards')
        guards_mod.GlobalStateGuard = type('GlobalStateGuard', (), {})
        sys.modules['torch._C._dynamo.guards'] = guards_mod
        torch._C._dynamo.guards = guards_mod

    if not hasattr(torch._C._dynamo, 'eval_frame'):
        eval_frame_mod = types.ModuleType('torch._C._dynamo.eval_frame')
        eval_frame_mod.set_context_frame = lambda *args, **kwargs: None
        sys.modules['torch._C._dynamo.eval_frame'] = eval_frame_mod
        torch._C._dynamo.eval_frame = eval_frame_mod
except Exception as e:
    pass

for attr in ['SymInt', 'SymFloat', 'SymBool']:
    if not hasattr(torch, attr):
        setattr(torch, attr, int)

for fn in ['sym_float', 'sym_int', 'sym_bool']:
    if not hasattr(torch, fn):
        setattr(torch, fn, lambda x, *args, **kwargs: float(x) if 'float' in fn else int(x))

try:
    import torch.library
    if not hasattr(torch.library, 'get_ctx'):
        def _dummy_get_ctx(*args, **kwargs):
            return None
        torch.library.get_ctx = _dummy_get_ctx
        
    if not hasattr(torch.library, 'register_fake'):
        def _dummy_register_fake(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        torch.library.register_fake = _dummy_register_fake
except Exception as e:
    pass

# Patch torch._utils_internal for missing log_trace_structured_event
try:
    import torch._utils_internal as _ui
    if not hasattr(_ui, 'log_trace_structured_event'):
        def _dummy_log_trace(*args, **kwargs):
            pass
        setattr(_ui, 'log_trace_structured_event', _dummy_log_trace)
except Exception as e:
    pass

try:
    types_path = "C:/Users/DHAARM PRASATH M/anaconda3/envs/PTBXL/lib/python3.10/site-packages/torchgen/api/types/types.py"
    if os.path.exists(types_path):
        with open(types_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        modified = False
        for line in lines:
            if ("DimVector" in line or "Float8" in line or "DeviceIndex" in line) and not line.strip().startswith("#"):
                new_lines.append("# [Auto-patched mismatch] " + line)
                modified = True
            else:
                new_lines.append(line)
                
        if modified:
            with open(types_path, "w") as f:
                f.writelines(new_lines)
except Exception as e:
    print(f"Auto-patch error: {e}")

# --------------------------------------------------
# STANDARD IMPORTS
# --------------------------------------------------
import copy
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import timm
from captum.attr import IntegratedGradients, Saliency, Occlusion
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# --------------------------------------------------
# 1. DIRECTORY CONFIGURATION & CUDA OPTIMIZATIONS
# --------------------------------------------------
DATA_DIR = Path(r"C:\Users\DHAARM PRASATH M\Documents\GPU_Check\WCEBleedGen")
SPLIT_DIR = DATA_DIR / "split"
BASE_OUTPUT_DIR = r"C:/Users/DHAARM PRASATH M/Documents/GPU_Check/XAI_WCE_DINOv2_SOA_Elman"

DIR_METRICS = os.path.join(BASE_OUTPUT_DIR, "metrics")
DIR_OPTIM = os.path.join(BASE_OUTPUT_DIR, "optimization")
DIR_XAI = os.path.join(BASE_OUTPUT_DIR, "xai_maps")
DIR_CHECKPOINTS = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
DIR_SOTA = os.path.join(BASE_OUTPUT_DIR, "sota_comparison")

for d in [DIR_METRICS, DIR_OPTIM, DIR_XAI, DIR_CHECKPOINTS, DIR_SOTA]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 1               # Configured to 1 epoch for quick validation (change to 100 for full convergence)
SOTA_BENCHMARK_EPOCHS = 20  # Baseline models training epochs
SOA_ITERATIONS = 30
SOA_POP_SIZE = 20
ELMAN_HIDDEN_DIM = 64
NUM_WORKERS = 0
XAI_SAMPLES = 12
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CUDA hardware acceleration, TF32, and AMP settings
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(
        f"CUDA {torch.version.cuda} Active | Hardware: {gpu_name} "
        f"({gpu_mem:.2f} GB VRAM) | TF32 Enabled"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))


def autocast_context():
    return torch.amp.autocast(
        device_type=device.type,
        enabled=(device.type == "cuda"),
    )

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# --------------------------------------------------
# 2. DATASETS & LOADERS
# --------------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        split = pd.read_csv(csv_path)
        required_columns = {"path", "label"}
        missing_columns = required_columns.difference(split.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_path} is missing required columns: {sorted(missing_columns)}"
            )
        if split.empty:
            raise ValueError(f"No image records found in {csv_path}")

        labels = pd.to_numeric(split["label"], errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError(f"Labels in {csv_path} must be 0 or 1")

        self.samples = []
        for image_path, label in zip(split["path"], labels):
            label = int(label)
            class_name = "bleeding" if label == 1 else "non_bleeding"
            csv_class_name = "bleeding" if label == 1 else "non-bleeding"
            source_parts = str(image_path).replace("\\", "/").split("/")
            if csv_class_name not in source_parts:
                raise ValueError(
                    f"Label {label} does not match the class in CSV path: {image_path}"
                )

            local_path = DATA_DIR / class_name / "images" / Path(source_parts[-1])
            if not local_path.is_file():
                raise FileNotFoundError(
                    f"Image listed in {csv_path} was not found: {local_path}"
                )
            self.samples.append((local_path, label))

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


train_ds = CSVDataset(SPLIT_DIR / "train.csv", transform=train_transform)
val_ds = CSVDataset(SPLIT_DIR / "val.csv", transform=eval_transform)
test_ds = CSVDataset(SPLIT_DIR / "test.csv", transform=eval_transform)

class_names = ["non_bleeding", "bleeding"]
num_classes = len(class_names)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

labels = np.array(train_ds.targets)
counts = np.bincount(labels, minlength=num_classes)
class_weights = len(labels) / (num_classes * np.maximum(counts, 1))
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

# --------------------------------------------------
# 3. ELMAN RECURRENT NEURAL NETWORK & ARCHITECTURE
# --------------------------------------------------
class ElmanNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ElmanNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = nn.RNNCell(input_dim, hidden_dim, nonlinearity='tanh')
        self.dropout = nn.Dropout(0.2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        h = self.rnn_cell(x, h)
        h = self.dropout(h)
        return self.fc_out(h)

    def get_flat_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat_params):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p))
            offset += numel

class CompositeModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(CompositeModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

dinov2_backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True).to(device)
feature_dim = dinov2_backbone.num_features

elman_classifier = ElmanNN(input_dim=feature_dim, hidden_dim=ELMAN_HIDDEN_DIM, output_dim=num_classes).to(device)
proposed_model = CompositeModel(dinov2_backbone, elman_classifier).to(device)

# --------------------------------------------------
# 4. TRAINING PIPELINE (WITH AUTOMATIC MIXED PRECISION)
# --------------------------------------------------
print(f"\n[Phase 1] Training Proposed Architecture for {EPOCHS} Epoch(s) using AMP...")
optimizer = torch.optim.AdamW(proposed_model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

history_proposed = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_f1 = -1.0
best_model_weights = copy.deepcopy(proposed_model.state_dict())

for epoch in range(EPOCHS):
    proposed_model.train()
    tot_loss, corr, tot = 0.0, 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        
        # Mixed Precision Forward & Backward Pass
        with autocast_context():
            out = proposed_model(imgs)
            loss = criterion(out, lbls)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(proposed_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item() * imgs.size(0)
        corr += (out.argmax(dim=1) == lbls).sum().item()
        tot += lbls.size(0)

    scheduler.step()
    tr_loss, tr_acc = tot_loss / tot, corr / tot

    proposed_model.eval()
    v_loss, v_corr, v_tot = 0.0, 0, 0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            with autocast_context():
                out = proposed_model(imgs)
                v_loss += criterion(out, lbls).item() * imgs.size(0)
            preds = out.argmax(dim=1)
            v_corr += (preds == lbls).sum().item()
            v_tot += lbls.size(0)
            val_preds.extend(preds.cpu().tolist())
            val_targets.extend(lbls.cpu().tolist())

    val_loss, val_acc = v_loss / v_tot, v_corr / v_tot
    val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

    history_proposed["train_loss"].append(tr_loss)
    history_proposed["val_loss"].append(val_loss)
    history_proposed["train_acc"].append(tr_acc)
    history_proposed["val_acc"].append(val_acc)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_weights = copy.deepcopy(proposed_model.state_dict())

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

proposed_model.load_state_dict(best_model_weights)

# --------------------------------------------------
# 5. SEEKER OPTIMIZATION ALGORITHM (SOA)
# --------------------------------------------------
print("\n[Phase 2] Optimizing Elman Classifier via Seeker Optimization Algorithm (SOA)...")
def extract_feature_cache(loader, backbone_net):
    feats, targets = [], []
    backbone_net.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            with autocast_context():
                feats.append(backbone_net(imgs))
            targets.append(lbls.to(device))
    return torch.cat(feats, dim=0), torch.cat(targets, dim=0)

train_feats, train_targets = extract_feature_cache(train_loader, dinov2_backbone)

def evaluate_fitness(flat_w):
    elman_classifier.set_flat_params(flat_w)
    elman_classifier.eval()
    with torch.no_grad():
        with autocast_context():
            logits = elman_classifier(train_feats)
            loss = criterion(logits, train_targets).item()
    return loss

num_params = elman_classifier.get_flat_params().numel()
base_w = elman_classifier.get_flat_params().clone()

pop = torch.randn(SOA_POP_SIZE, num_params, device=device) * 0.02 + base_w.unsqueeze(0)
fitness = torch.tensor([evaluate_fitness(pop[i]) for i in range(SOA_POP_SIZE)], device=device)

best_idx = torch.argmin(fitness)
best_fitness = fitness[best_idx].item()
best_solution = pop[best_idx].clone()

pbest = pop.clone()
pbest_fit = fitness.clone()
soa_fitness_history = [best_fitness]

w_max, w_min = 0.9, 0.4
for t in range(SOA_ITERATIONS):
    weight = w_max - (w_max - w_min) * (t / SOA_ITERATIONS)
    for i in range(SOA_POP_SIZE):
        d_ego = torch.sign(pbest[i] - pop[i])
        d_alt = torch.sign(best_solution - pop[i])
        rand_ego, rand_alt = random.random(), random.random()
        direction = torch.sign(weight * (rand_ego * d_ego + rand_alt * d_alt) + (1.0 - weight) * torch.randn_like(pop[i]))
        step_len = torch.abs(torch.randn_like(pop[i])) * (0.005 * (1.0 - t / SOA_ITERATIONS))
        pop[i] = pop[i] + step_len * direction

        current_fit = evaluate_fitness(pop[i])
        fitness[i] = current_fit
        if current_fit < pbest_fit[i]:
            pbest_fit[i] = current_fit
            pbest[i] = pop[i].clone()
        if current_fit < best_fitness:
            best_fitness = current_fit
            best_solution = pop[i].clone()

    soa_fitness_history.append(best_fitness)
    if (t + 1) % 5 == 0 or t == 0:
        print(f"SOA Iteration {t+1:02d}/{SOA_ITERATIONS:02d} | Best Fitness: {best_fitness:.5f}")

elman_classifier.set_flat_params(best_solution)
torch.save(proposed_model.state_dict(), os.path.join(DIR_CHECKPOINTS, "proposed_soa_elman_dino.pth"))

# --------------------------------------------------
# 6. SOTA BENCHMARK COMPARISONS
# --------------------------------------------------
print("\n[Phase 3] Benchmarking Against SOTA Models...")

sota_model_factories = {
    "Proposed (DINOv2+Elman+SOA)": None,
    "Standard DINOv2": lambda: timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=num_classes,
        dynamic_img_size=True,
    ).to(device),
    "Swin-Transformer (Tiny)": lambda: timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes,
    ).to(device),
    "ConvNeXtV2 (Tiny)": lambda: timm.create_model(
        "convnextv2_tiny",
        pretrained=True,
        num_classes=num_classes,
    ).to(device),
    "ResNet-50": lambda: timm.create_model(
        "resnet50",
        pretrained=True,
        num_classes=num_classes,
    ).to(device),
    "DenseNet-121": lambda: timm.create_model(
        "densenet121",
        pretrained=True,
        num_classes=num_classes,
    ).to(device),
}

def train_sota(m_name, model_inst):
    if m_name == "Proposed (DINOv2+Elman+SOA)":
        return
    opt = torch.optim.AdamW(model_inst.parameters(), lr=1e-4, weight_decay=0.01)
    for epoch in range(SOTA_BENCHMARK_EPOCHS):
        model_inst.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            with autocast_context():
                out = model_inst(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

def evaluate_model(model_inst):
    model_inst.eval()
    y_t, y_p = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            with autocast_context():
                preds = model_inst(imgs).argmax(dim=1).cpu().tolist()
            y_p.extend(preds)
            y_t.extend(lbls.tolist())
    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, average="macro", zero_division=0)
    rec = recall_score(y_t, y_p, average="macro", zero_division=0)
    f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
    return acc, prec, rec, f1, y_t, y_p

sota_results = []
test_predictions_dict = {}

for name, model_factory in sota_model_factories.items():
    print(f"Evaluating {name}...")
    model_inst = proposed_model if model_factory is None else model_factory()
    train_sota(name, model_inst)
    acc, prec, rec, f1, y_t, y_p = evaluate_model(model_inst)
    test_predictions_dict[name] = (y_t, y_p)
    sota_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Macro F1": f1
    })
    if model_factory is not None:
        del model_inst
        if device.type == "cuda":
            torch.cuda.empty_cache()

df_sota = pd.DataFrame(sota_results)
df_sota.to_csv(os.path.join(DIR_SOTA, "sota_comparison_table.csv"), index=False)
print("\n--- SOTA BENCHMARK SUMMARY ---")
print(df_sota.to_string(index=False))

# SOTA Bar Chart
metrics_to_plot = ["Accuracy", "Precision", "Recall", "Macro F1"]
x = np.arange(len(df_sota["Model"]))
width = 0.2

plt.figure(figsize=(13, 6))
for i, m in enumerate(metrics_to_plot):
    plt.bar(x + i * width, df_sota[m], width, label=m)

plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Comparative Performance Analysis with SOTA Architectures")
plt.xticks(x + width * 1.5, df_sota["Model"], rotation=15)
plt.ylim(0.0, 1.05)
plt.legend(loc="lower right")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(DIR_SOTA, "sota_metrics_barchart.png"), dpi=200)
plt.close()

# --------------------------------------------------
# 7. OPTIMIZATION & PERFORMANCE CURVES
# --------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(soa_fitness_history, marker='o', color='teal', linewidth=2)
plt.title("Seeker Optimization Algorithm (SOA) Convergence")
plt.xlabel("Iteration")
plt.ylabel("Fitness (Cross-Entropy Loss)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(DIR_OPTIM, "soa_convergence_curve.png"), dpi=200)
plt.close()

plt.figure(figsize=(12, 4.5))
plt.subplot(1, 2, 1)
plt.plot(history_proposed["train_loss"], label="Train", color="crimson")
plt.plot(history_proposed["val_loss"], label="Val", color="orange")
plt.title(f"Proposed Model Loss ({EPOCHS} Epochs)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(history_proposed["train_acc"], label="Train", color="royalblue")
plt.plot(history_proposed["val_acc"], label="Val", color="forestgreen")
plt.title(f"Proposed Model Accuracy ({EPOCHS} Epochs)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(DIR_METRICS, "loss_and_accuracy_curves.png"), dpi=200)
plt.close()

y_true_prop, y_pred_prop = test_predictions_dict["Proposed (DINOv2+Elman+SOA)"]
cm = confusion_matrix(y_true_prop, y_pred_prop, labels=list(range(num_classes)))
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.xticks(range(num_classes), class_names, rotation=20)
plt.yticks(range(num_classes), class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Proposed Architecture)")
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig(os.path.join(DIR_METRICS, "test_confusion_matrix.png"), dpi=200)
plt.close()

# --------------------------------------------------
# 8. XAI ATTRIBUTIONS (INTEGRATED GRADIENTS, SALIENCY, OCCLUSION)
# --------------------------------------------------
print("\n[Phase 4] Generating XAI Heatmaps...")
ig = IntegratedGradients(proposed_model)
saliency = Saliency(proposed_model)
occlusion = Occlusion(proposed_model)

def denormalize(img):
    m = torch.tensor(mean, device=img.device).view(3, 1, 1)
    st = torch.tensor(std, device=img.device).view(3, 1, 1)
    return (img * st + m).clamp(0, 1)

def normalize_map(attr):
    attr = attr.detach().abs().sum(dim=1, keepdim=True)
    attr = attr - attr.amin(dim=(2, 3), keepdim=True)
    return attr / (attr.amax(dim=(2, 3), keepdim=True) + 1e-8)

proposed_model.eval()
saved = 0
for imgs, targets in test_loader:
    imgs = imgs.to(device)
    with torch.no_grad():
        with autocast_context():
            logits = proposed_model(imgs)
            probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    for i in range(imgs.size(0)):
        input_img = imgs[i:i+1].detach()
        target_cls = int(preds[i].item())

        ig_attr = ig.attribute(input_img, baselines=torch.zeros_like(input_img), target=target_cls, n_steps=20, internal_batch_size=2)
        sal_attr = saliency.attribute(input_img, target=target_cls, abs=True)
        occ_attr = occlusion.attribute(
            input_img, baselines=0, target=target_cls,
            sliding_window_shapes=(3, 32, 32), strides=(3, 16, 16),
            perturbations_per_eval=4
        )

        maps = [normalize_map(a)[0, 0].cpu().numpy() for a in (ig_attr, sal_attr, occ_attr)]
        orig = denormalize(input_img[0]).permute(1, 2, 0).cpu().numpy()
        titles = ["Integrated Gradients", "Saliency Map", "Occlusion Map"]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(orig)
        axes[0].set_title("Original Image")
        for ax, heat, t_name in zip(axes[1:], maps, titles):
            ax.imshow(orig)
            ax.imshow(heat, cmap="jet", alpha=0.45)
            ax.set_title(t_name)
        fig.suptitle(
            f"Predicted: {class_names[target_cls]} | Prob: {probs[i, target_cls].item():.3f} | True: {class_names[int(targets[i])]}",
            fontsize=11
        )
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(DIR_XAI, f"xai_sample_{saved:04d}.png"), dpi=180)
        plt.close(fig)

        saved += 1
        if saved >= XAI_SAMPLES:
            break
    if saved >= XAI_SAMPLES:
        break

print(f"\nExecution finished successfully. Outputs organized under: {BASE_OUTPUT_DIR}")