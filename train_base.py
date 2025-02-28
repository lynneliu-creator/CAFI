import torch
import os
import sys
import argparse
import numpy as np
sys.path.append('./')
from utils_attack import pgd
import torch.nn.functional as F
from trainer_base import SOLVER
from utils import *
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
base = importr('base')
rvinecop = importr('rvinecopulib')


class AT(SOLVER):
    def __init__(self, args):
        super().__init__(args)
        self.path = os.path.join('trained_model', args.dataset.lower(), 'nt', f'beta_{args.beta}')
        os.makedirs(self.path, exist_ok=True)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def loss(self, image, delta, label):
        loss_clean = F.cross_entropy(self.network(image)[0], label)
        loss_robust = F.cross_entropy(self.network(image + delta)[0], label)
        loss_total = (1 - self.beta) * loss_clean + self.beta * loss_robust
        return loss_total

    def train_epoch(self):
        self.network.train()
        for batch_idx, (image, label) in enumerate(self.train_loader):
            image, label = image.to(self.device), label.to(self.device)
            self.optim.zero_grad()
            delta = pgd(self.args, self.network, image, label)
            loss_total = self.loss(image, delta, label)
            loss_total.backward()
            self.optim.step()

def main():
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0)  # beta=0 natural training; beta=1 adversarial training
    parser.add_argument("--dataset", type=str, default="CIFAR-10")  # "CIFAR-100", "TinyImageNet"
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--out_dim", type=int, default=15)
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--tensorboard", type=bool, default=False)
    
    # Attack parameters
    parser.add_argument("--epsilon", type=float, default=0.031)
    parser.add_argument("--alpha", type=float, default=0.007)
    parser.add_argument("--num_iter", type=int, default=10)
    
    args = parser.parse_args()
    print(args)

    if not (0 <= args.beta <= 1):
        sys.exit('Wrong beta range. beta should be in [0, 1]')
    
    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1]

    torch.manual_seed(args.seed)
    trainer = AT(args)

    for epoch in range(1, args.epochs + 1):
        trainer.adjust_learning_rate(epoch)
        print(f'Epoch: {epoch}, Learning Rate: {round(trainer.optim.param_groups[0]["lr"], 5)}')
        
        trainer.train_epoch()
        trainer.train_eval(epoch)
        trainer.test_eval(epoch)

    torch.save(trainer.network.state_dict(), os.path.join(trainer.path, f'epoch_{epoch}_v2.pth'))

if __name__ == '__main__':
    main()
