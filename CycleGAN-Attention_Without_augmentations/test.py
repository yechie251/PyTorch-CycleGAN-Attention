
#!/usr/bin/python3
# test_full.py — generate outputs (A->B, B->A) + compute FID & MiFID

import argparse, sys, os, math
from pathlib import Path
from PIL import Image
from torchvision.utils import save_image, make_grid

import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from torchvision.models import inception_v3

from models import Generator
from datasets import ImageDataset
from MIFID import compute_fid_mifid

torch.backends.cudnn.benchmark = True


def load_state_safely(model: torch.nn.Module, path: str, device):
    """Load state_dict, handling DataParallel prefixes and map_location."""
    try:
        sd = torch.load(path, map_location=device)  # if supported in your version: , weights_only=True
    except TypeError:
        sd = torch.load(path, map_location=device)
    if isinstance(sd, dict) and len(sd) > 0:
        first = next(iter(sd.keys()))
        if first.startswith("module."):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='batch size')
    parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root dir of dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='input channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output channels')
    parser.add_argument('--size', type=int, default=256, help='square size (HxW)')
    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--n_cpu', type=int, default=0, help='DataLoader workers (Windows→0)')
    parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B checkpoint')
    parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A checkpoint')
    # Metrics:
    parser.add_argument('--mifid', type=str, default=None, choices=['AtoB','BtoA','both'],
                        help='Compute FID & MiFID after generating images')
    parser.add_argument('--mifid_max', type=int, default=1000, help='Limit images per side for quick eval')
    parser.add_argument('--tau', type=float, default=0.2, help='MiFID penalty threshold')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: CUDA device is available; consider adding --cuda")

    device = torch.device('cuda' if (opt.cuda and torch.cuda.is_available()) else 'cpu')

    # ---------- Models ----------
    netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
    netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)
    load_state_safely(netG_A2B, opt.generator_A2B, device)
    load_state_safely(netG_B2A, opt.generator_B2A, device)
    netG_A2B.eval(); netG_B2A.eval()

    # ---------- Data ----------
    tf = [
        transforms.Resize((opt.size, opt.size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        ImageDataset(opt.dataroot, transforms_=tf, mode='test'),
        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu
    )

    # ---------- Outputs ----------
    os.makedirs('output/A', exist_ok=True)  # fake_A: Monet->Photo
    os.makedirs('output/B', exist_ok=True)  # fake_B: Photo->Monet
    os.makedirs('output/AtoB_compare', exist_ok=True)
    os.makedirs('output/BtoA_compare', exist_ok=True)
    with torch.no_grad():
        idx = 0
        total = len(dataloader)
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            fake_B = (netG_A2B(real_A) + 1.0) * 0.5  # -> output/B
            fake_A = (netG_B2A(real_B) + 1.0) * 0.5  # -> output/A

            # Save per-sample (not a grid), supports batchSize>1
            bsz = fake_A.size(0)
            for b in range(bsz):
                idx += 1
                save_image(fake_A, f'output/A/{i+1:04d}.png')
                save_image(fake_B, f'output/B/{i+1:04d}.png')

                combined_AB = torch.cat((0.5*(real_A.data + 1.0), fake_B), 0)
                grid_AB = make_grid(combined_AB, nrow=2)
                save_image(grid_AB, f'output/AtoB_compare/{i+1:04d}.png')

                combined_BA = torch.cat((0.5*(real_B.data + 1.0), fake_A), 0)
                grid_BA = make_grid(combined_BA, nrow=2)
                save_image(grid_BA, f'output/BtoA_compare/{i+1:04d}.png')


            sys.stdout.write(f'\rGenerated images {i+1:04d} of {total:04d}')
        sys.stdout.write('\n')

    # ---------- Metrics (optional) ----------
    if opt.mifid is not None:
        device_str = 'cuda' if (device.type == 'cuda') else 'cpu'
        print("\n[Metrics] device:", device_str)

        def run_eval(direction):
            if direction == 'AtoB':
                gen_dir    = 'output/B'  # fake_B (Photo->Monet)
                real_test  = os.path.join(opt.dataroot, 'test',  'B')
                real_train = os.path.join(opt.dataroot, 'train', 'B')
            else:  # 'BtoA'
                gen_dir    = 'output/A'  # fake_A (Monet->Photo)
                real_test  = os.path.join(opt.dataroot, 'test',  'A')
                real_train = os.path.join(opt.dataroot, 'train', 'A')

            print(f"\n[Metrics] {direction}: gen_dir={gen_dir}")
            fid, M, mifid_div, mifid_pen = compute_fid_mifid(
                gen_dir, real_test, real_train,
                device=device_str, max_n=opt.mifid_max, tau=opt.tau, batch_size=32)
            print(f"FID ({direction}): {fid:.4f}")
            print(f"M  ({direction}) mean(1 - max cos to TRAIN): {M:.4f}  (higher = less memorization)")
            print(f"MiFID_div ({direction}) = FID/M: {mifid_div:.4f}")
            print(f"MiFID_pen ({direction}) @tau={opt.tau}: {mifid_pen:.4f}")

        if opt.mifid in ('AtoB', 'both'):
            run_eval('AtoB')
        if opt.mifid in ('BtoA', 'both'):
            run_eval('BtoA')

if __name__ == '__main__':
    main()
