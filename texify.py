# pip install -r requirements.txt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import glob, json, argparse, yaml

from PIL import Image
from pathlib import Path
from typing import List, Dict
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------- Vocab --------------------------
def load_vocab(vocab_path: str):
    with open(vocab_path, "r") as f:
        y = yaml.safe_load(f)

    # Canonical class list (order defines indices)
    classes = [c["name"] for c in y["classes"]]

    # Canonical -> LaTeX token
    token_map = {c["name"]: c["token"] for c in y["classes"]}

    # Case-insensitive alias lookup: lower(alias) -> canonical name
    alias_map = {}
    for c in y["classes"]:
        canon = c["name"]
        alias_map[canon.lower()] = canon
        for a in c.get("aliases", []):
            alias_map[a.lower()] = canon

    return y.get("version", 1), classes, token_map, alias_map

# -------------------------- Filename → label --------------------------
def parse_label_from_filename(stem: str) -> str:
    """
    Example: 'alpha_001' -> 'alpha'
    We return LOWERCASE so it matches alias_map keys.
    """
    base = stem.split("_", 1)[0]
    return base.lower()

# -------------------------- Dataset --------------------------
class SymbolsDataset(Dataset):
    def __init__(self, root, classes: List[str], alias_map: Dict[str, str],
                 img_size=128, train=True):
        root = Path(root)
        # Restrict to JPG/JPEG as per your data; add more exts if needed
        exts = ("*.jpg", "*.jpeg", "*.png", ".JPG", "*.JPEG", "*.PNG")
        self.paths = []
        for ext in exts:
            self.paths.extend(glob.glob(str(root / ext)))
            self.paths.extend(glob.glob(str(root / "*" / ext)))  # allow one subfolder level

        if not self.paths:
            raise FileNotFoundError(f"No images under {root}")

        # Canonical ordering from YAML
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self.samples, self.labels_for_weights = [], []
        for p in self.paths:
            stem = Path(p).stem
            key_lower = parse_label_from_filename(stem)  # lowercased
            canon = alias_map.get(key_lower)             # canonical name or None
            if canon is None or canon not in self.class_to_idx:
                # Skip files whose label isn't listed in vocab.yaml
                continue
            y = self.class_to_idx[canon]
            self.samples.append((p, y))
            self.labels_for_weights.append(y)

        # Transforms
        aug = transforms.RandomApply([
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), shear=5)
        ], p=0.9)
        jitter = transforms.RandomApply([transforms.ColorJitter(0.15, 0.15)], p=0.5)
        base = [
            transforms.Grayscale(1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        self.tfm = transforms.Compose(([aug, jitter] if train else []) + base)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("L")
        x = self.tfm(img)
        return x, y

    def make_sampler(self):
        counts = np.bincount(np.array(self.labels_for_weights))
        weights = 1.0 / np.maximum(counts, 1)
        sample_w = [weights[y] for _, y in self.samples]
        return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

# -------------------------- Models --------------------------
class Encoder(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.body = nn.Sequential(*list(m.children())[:-1])  # B,512,1,1
        self.proj = nn.Linear(512, emb_dim)
    def forward(self, x):
        f = self.body(x).squeeze(-1).squeeze(-1)  # B,512
        z = self.proj(f)
        return F.normalize(z, dim=1)

class CosineHead(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.W = nn.Parameter(F.normalize(torch.randn(num_classes, emb_dim), dim=1))
        self.scale = nn.Parameter(torch.tensor(10.0))
    def forward(self, z):  # z: (B,D) normalized
        return self.scale * (z @ self.W.T)
    def expand(self, k):
        with torch.no_grad():
            newW = F.normalize(torch.randn(k, self.W.size(1)), dim=1)
        self.W = nn.Parameter(torch.cat([self.W.data, newW], dim=0))

# -------------------------- Train / Eval --------------------------
def train_epoch(encoder, head, dl, opt, ce):
    encoder.train(); head.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        z = encoder(x)
        logits = head(z)
        loss = ce(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def eval_epoch(encoder, head, dl):
    encoder.eval(); head.eval()
    total, correct = 0, 0
    y_true, y_pred = [], []
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        z = encoder(x)
        logits = head(z)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total   += x.size(0)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
    return correct / total, y_true, y_pred

def save_ckpt(path, encoder, head, classes, token_map, img_size, vocab_version):
    torch.save({
        "encoder": encoder.state_dict(),
        "head": head.state_dict(),
        "classes": classes,
        "token_map": token_map,
        "img_size": img_size,
        "vocab_version": vocab_version,
        "arch": {"emb_dim": head.W.size(1)}
    }, path)

def load_ckpt(path):
    ck = torch.load(path, map_location="cpu")
    enc = Encoder(emb_dim=ck["arch"]["emb_dim"])
    head = CosineHead(emb_dim=ck["arch"]["emb_dim"], num_classes=len(ck["classes"]))
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    return ck, enc, head

# -------------------------- CLI Commands --------------------------
def cmd_train(args):
    vver, classes, token_map, alias_map = load_vocab(args.vocab)
    img_size = args.img_size

    train_ds = SymbolsDataset(Path(args.data) / "train", classes, alias_map, img_size, train=True)
    test_ds  = SymbolsDataset(Path(args.data) / "test",  classes, alias_map, img_size, train=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=train_ds.make_sampler(),
                          num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                          num_workers=2, pin_memory=True)

    encoder = Encoder(emb_dim=args.emb).to(DEVICE)
    head    = CosineHead(emb_dim=args.emb, num_classes=len(classes)).to(DEVICE)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
    ce  = nn.CrossEntropyLoss()

    best, ckpt = 0.0, args.ckpt
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(encoder, head, train_dl, opt, ce)
        va_acc, y_true, y_pred = eval_epoch(encoder, head, test_dl)
        print(f"Epoch {ep:02d}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}  loss={tr_loss:.4f}")
        if va_acc > best:
            best = va_acc
            save_ckpt(ckpt, encoder, head, classes, token_map, img_size, vver)
            print(f"Saved {ckpt} (val_acc={best:.4f})")

    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print(confusion_matrix(y_true, y_pred))

def cmd_expand(args):
    # Load existing
    ck, enc, head = load_ckpt(args.ckpt)
    old_classes   = ck["classes"]
    old_token_map = ck["token_map"]
    img_size      = ck["img_size"]
    vocab_version_old = ck.get("vocab_version", 1)

    # New vocab (superset)
    vver, new_classes, new_token_map, alias_map = load_vocab(args.vocab)
    if vver < vocab_version_old:
        print("Warning: vocab version older than checkpoint.")

    added = [c for c in new_classes if c not in old_classes]
    if not added:
        print("No new classes in vocab. Nothing to do.")
        return

    # Expand head
    head = CosineHead(emb_dim=head.W.size(1), num_classes=len(old_classes))
    head.load_state_dict(ck["head"])
    head.expand(len(added))

    classes = old_classes + added
    token_map = {**old_token_map, **new_token_map}

    train_ds = SymbolsDataset(Path(args.data) / "train", classes, alias_map, img_size, train=True)
    test_ds  = SymbolsDataset(Path(args.data) / "test",  classes, alias_map, img_size, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=train_ds.make_sampler(),
                          num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                          num_workers=2, pin_memory=True)

    # Freeze encoder, train head only
    enc.to(DEVICE).eval()
    for p in enc.parameters(): p.requires_grad = False
    head.to(DEVICE).train()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    ce  = nn.CrossEntropyLoss()

    best, out_ckpt = 0.0, args.out
    for ep in range(1, args.epochs + 1):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                z = enc(x)
            opt.zero_grad()
            logits = head(z)
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct  += (logits.argmax(1) == y).sum().item()
            total    += x.size(0)
        tr_acc = correct / total
        va_acc, _, _ = eval_epoch(enc, head, test_dl)
        print(f"[expand] Epoch {ep:02d} train_acc={tr_acc:.4f} val_acc={va_acc:.4f} loss={loss_sum/total:.4f}")
        if va_acc > best:
            best = va_acc
            torch.save({
                "encoder": enc.state_dict(),
                "head": head.state_dict(),
                "classes": classes,
                "token_map": token_map,
                "img_size": img_size,
                "vocab_version": vver,
                "arch": {"emb_dim": head.W.size(1)}
            }, out_ckpt)
            print(f"Saved expanded {out_ckpt} (val_acc={best:.4f})")

def cmd_predict(args):
    ck, enc, head = load_ckpt(args.ckpt)
    classes = ck["classes"]
    token_map = ck["token_map"]
    img_size = ck["img_size"]
    enc.to(DEVICE).eval(); head.to(DEVICE).eval()

    tfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    for p in args.images:
        img = Image.open(p).convert("L")
        x = tfm(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z = enc(x)
            logits = head(z)
            idx = int(logits.argmax(1))
            name = classes[idx]
            latex = token_map.get(name, name)
            print(json.dumps({"path": p, "class": name, "latex": latex}))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    ap_train = sub.add_parser("train")
    ap_train.add_argument("--data", required=True)       # folder with train/ and test/
    ap_train.add_argument("--vocab", required=True)      # vocab.yaml
    ap_train.add_argument("--ckpt", default="nimble.pt")
    ap_train.add_argument("--img_size", type=int, default=128)
    ap_train.add_argument("--emb", type=int, default=512)
    ap_train.add_argument("--batch", type=int, default=64)
    ap_train.add_argument("--epochs", type=int, default=15)
    ap_train.add_argument("--lr", type=float, default=3e-4)
    ap_train.set_defaults(func=cmd_train)

    # expand
    ap_exp = sub.add_parser("expand")
    ap_exp.add_argument("--data", required=True)
    ap_exp.add_argument("--vocab", required=True)
    ap_exp.add_argument("--ckpt", required=True)
    ap_exp.add_argument("--out",  default="nimble_expanded.pt")
    ap_exp.add_argument("--epochs", type=int, default=6)
    ap_exp.add_argument("--batch", type=int, default=64)
    ap_exp.add_argument("--lr", type=float, default=5e-4)
    ap_exp.set_defaults(func=cmd_expand)

    # predict
    ap_pred = sub.add_parser("predict")
    ap_pred.add_argument("--ckpt", required=True)
    ap_pred.add_argument("images", nargs="+")
    ap_pred.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()