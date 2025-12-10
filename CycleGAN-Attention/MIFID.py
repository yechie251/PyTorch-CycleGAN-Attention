
import argparse, sys, os, math
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image, make_grid
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import inception_v3



IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

class ImgFolder(Dataset):
    """Loads images from a folder for metrics (returns tensors in [0,1], size 299x299)."""
    def __init__(self, root: str, max_n: int | None = None):
        p = Path(root)
        self.paths = sorted([x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS])
        if max_n is not None:
            self.paths = self.paths[:max_n]
            self.tf = T.Compose([
            T.Resize((299, 299)),
            T.PILToTensor(), 
            ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return self.tf(img)

@torch.no_grad()
def extract_feats(dset: Dataset, device='cuda', batch_size=32):
    """Inception-v3 2048d features for memorization check."""
    m = inception_v3(weights='IMAGENET1K_V1', aux_logits=True, transform_input=False)
    m.fc = torch.nn.Identity()
    m.eval().to(device)

    feats = []
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=0)
    for x in loader:
        x = x.to(device).float() / 255.0  
        f = m(x)  # [B, 2048]
        feats.append(f.detach())

    return torch.cat(feats, 0)  # [N, 2048]

@torch.no_grad()
def memorization_score(gen_feats: torch.Tensor, train_feats: torch.Tensor, chunk=2048):
    """Mean(1 - max cosine sim) between each generated feature and nearest TRAIN target feature."""
    g = torch.nn.functional.normalize(gen_feats, dim=1)
    t = torch.nn.functional.normalize(train_feats, dim=1)
    max_sims = []
    n = g.size(0)
    step = max(1, math.ceil(n / max(1, n // chunk)))
    for i in range(0, n, step):
        gi = g[i:i+step]           # [b,d]
        sims = gi @ t.T            # [b,Nt]
        max_sims.append(sims.max(dim=1).values)
    max_sims = torch.cat(max_sims, 0)
    M = (1.0 - max_sims).mean().item()   # bigger M = less memorization
    return M

def compute_fid_mifid(gen_dir: str, real_test: str, real_train: str,
                      device='cuda', max_n=1000, tau=0.2, batch_size=32):
    """Returns (FID, M, MiFID_div, MiFID_penalty)."""
    # FID
    ds_real = ImgFolder(real_test, max_n)
    ds_gen  = ImgFolder(gen_dir,  max_n)
    if len(ds_real) == 0 or len(ds_gen) == 0:
        raise RuntimeError(f'Empty folder for FID: real={real_test} ({len(ds_real)}), gen={gen_dir} ({len(ds_gen)})')
    fid = FrechetInceptionDistance(feature=2048).to(device)
    rl = DataLoader(ds_real, batch_size=batch_size, num_workers=0, shuffle=False)
    gl = DataLoader(ds_gen,  batch_size=batch_size, num_workers=0, shuffle=False)
    for x in rl: fid.update(x.to(device), real=True)
    for x in gl: fid.update(x.to(device), real=False)
    fid_score = float(fid.compute().cpu().item())

    # MiFID (memorization)
    ds_train = ImgFolder(real_train, max_n)
    if len(ds_train) == 0:
        raise RuntimeError(f'Empty TRAIN folder: {real_train}')
    gen_feats   = extract_feats(ds_gen,   device=device, batch_size=batch_size)
    train_feats = extract_feats(ds_train, device=device, batch_size=batch_size)
    M = memorization_score(gen_feats, train_feats)
    eps = 1e-8
    mifid_div = fid_score / max(M, eps)       # stricter when M is small (more memorization)
    penalty = (M/tau) if M < tau else 1.0
    mifid_pen = fid_score * penalty
    return fid_score, M, mifid_div, mifid_pen
