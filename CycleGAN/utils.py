
import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np

torch.backends.cudnn.benchmark = True

def gpu_check(step, models: dict, tensors: dict | None = None, every: int = 5000):
    """Assert & print that models/tensors are on CUDA every `every` steps."""
    if step % every != 0:
        return
    msgs = [f"GPU CHECK @ step {step}"]
    # check models
    for name, m in models.items():
        dev = next(m.parameters()).device
        msgs.append(f"{name}={dev}")
        assert dev.type == "cuda", f"{name} is on {dev}, expected cuda"
    # optional: check some tensors
    if tensors:
        for name, t in tensors.items():
            dev = t.device
            msgs.append(f"{name}={dev}")
            assert dev.type == "cuda", f"{name} is on {dev}, expected cuda"
    # memory snapshot (nice to have)
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        msgs.append(f"mem={used:.1f}MB/{total:.0f}MB")
        msgs.append(f"gpu={torch.cuda.get_device_name(0)}")
    print(" | ".join(msgs))
    
    
def _to_float(x):
    return x.detach().item() if torch.is_tensor(x) else float(x)

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                val = losses[loss_name]
                if torch.is_tensor(val):
                    val = val.detach().item()
                self.losses[loss_name] = float(val)
            else:
                self.losses[loss_name] += _to_float(losses[loss_name])


            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        try:
            if torch.cuda.is_available() and (batches_done % 5000 == 0):
                used = torch.cuda.memory_allocated() / (1024**2)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                # אם העברת תמונות ל-logger (images), נבדוק גם את הדיבייס של אחת מהן
                dev_str = ""
                if images:
                    any_tensor = next(iter(images.values()))
                    dev_str = f" | tensor_device={getattr(any_tensor, 'device', 'n/a')}"
                print(f"\n[GPU CHECK] step={batches_done} | gpu={torch.cuda.get_device_name(0)} | mem={used:.1f}MB/{total:.0f}MB{dev_str}")
        except Exception as e:
            print(f"\n[GPU CHECK] skipped ({e})")
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:
                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), 
                                                                    opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

