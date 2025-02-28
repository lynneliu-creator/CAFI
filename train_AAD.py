import torch
import os
import sys
import argparse
import numpy as np
sys.path.append('./')

from resnet import *
from preactresnet import *
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from TinyImageNet import TinyImageNetDataset

from utils_attack import pgd

import time
import datetime

from utils_vine import *
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
base = importr('base')
rvinecop = importr('rvinecopulib')

def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

class SOLVER:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.num_classes = args.num_classes
        self.out_dim = args.out_dim
        self.lr = args.lr
        self.gamma = args.gamma
        self.beta = args.beta
        self.epochs = args.epochs
        
        self._initialize_network()
        self._initialize_optimizer()
        self._initialize_data_loaders()


    def _initialize_network(self):
        network_map = {
            'ResNet18': ResNet18_twobranch,
            'PreActResNet18': PreActResNet18_twobranch_DenseV1 # for TinyImagenet
        }
        # tiny
        self.network =  PreActResNet18_twobranch_DenseV1(num_classes=self.num_classes, out_dim=self.out_dim,use_BN=False, along=True).to(self.device)
        # cifar
        # self.network =  ResNet18_twobranch(num_classes=self.num_classes, out_dim=self.out_dim).to(self.device)

    def _initialize_optimizer(self):
        self.optim = optim.SGD(self.network.parameters(), lr=self.lr, momentum=0.9, weight_decay=2e-4)

    def _initialize_data_loaders(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([transforms.ToTensor()])

        dataset_map = {
            'CIFAR-10': (datasets.CIFAR10, '/data/diska/liulin/data/MI/data/cifar10'),
            'CIFAR-100': (datasets.CIFAR100, '/data/diska/liulin/data/MI/data/cifar100'),
            'TinyImageNet': (TinyImageNetDataset, '/data/diska/liulin/data/MI/')
        }
        # cifar
        # dataset_class, dataset_root = dataset_map.get(self.args.dataset, (datasets.CIFAR10, '/data/diska/liulin/data/MI/data/cifar10'))
        # tiny
        dataset_class, dataset_root = dataset_map.get(self.args.dataset, (TinyImageNetDataset, '/data/diska/liulin/data/MI/'))
            # cifar
        if self.args.dataset=='CIFAR-10' or self.args.dataset=='CIFAR-100':
            trainset = dataset_class(root=dataset_root, train=True, download=True, transform=transform_train)
            testset = dataset_class(root=dataset_root, train=False, download=True, transform=transform_test)
        else:
            # tiny
            trainset = dataset_class(root=dataset_root,mode='train', download=False, transform=transform_train)
            testset = dataset_class(root=dataset_root,mode='val', download=False, transform=transform_test)

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def adjust_learning_rate(self, epoch):
        lr = self.lr
        if epoch >= 75:
            lr *= 0.1
        if epoch >= 90:
            lr *= 0.01
        if epoch >= 100:
            lr *= 0.001
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def train_eval(self, epoch):
        self.network.eval()
        train_loss = 0.
        correct = 0.
        cln_features = []

        with torch.no_grad():
            for X, label in self.train_loader:
                X, label = X.to(self.device), label.to(self.device)
                output, output_aux = self.network(X)
                if epoch == 1 or epoch == self.epochs:
                    cln_features.append(output_aux.detach().cpu())
                train_loss += F.cross_entropy(output, label).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()

        train_loss /= len(self.train_loader.dataset)
        accuracy = correct / len(self.train_loader.dataset)
        print(f"TRAIN LOSS: {round(train_loss, 5)} ACCURACY: {round(accuracy, 5)}")

        if epoch == 1 or epoch == self.epochs:
            self._save_vine_model(cln_features, epoch)


    def _save_vine_model(self, features, epoch):
        cln_features = torch.cat(features).numpy()
        print(f"\nfeatures.shape:\n{cln_features.shape}")
        copula_controls = base.list(family_set="tll", trunc_lvl=5, cores=4)
        cln_vine = rvinecop.vine(cln_features, copula_controls=copula_controls)
        # path = os.path.join('trained_model', self.args.dataset.lower(), 'AADat', f'gamma_{self.gamma}')
        path = os.path.join('trained_model', self.args.dataset.lower(), 'AADat', f'trades')
        os.makedirs(path, exist_ok=True)
        save_vinemodel(cln_vine, os.path.join(path, f'epoch{epoch}_vinemodel.pkl'))

    def test_eval(self, epoch):
        self.network.eval()
        test_loss = 0.
        correct = 0.

        with torch.no_grad():
            for X, label in self.test_loader:
                X, label = X.to(self.device), label.to(self.device)
                output, _ = self.network(X)
                test_loss += F.cross_entropy(output, label).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        print(f"TEST LOSS: {round(test_loss, 5)} ACCURACY: {round(accuracy, 5)}")




class AT(SOLVER):
    def __init__(self, args):
        super().__init__(args)
        self.path = os.path.join('trained_model', args.dataset.lower(), 'AADat', f'trades')
        os.makedirs(self.path, exist_ok=True)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        print(f"Using device: {self.device}")

    def loss(self, image, delta, label,sampled_py):
        # loss_clean = F.cross_entropy(self.network(image)[0], label)
        # loss_robust = F.cross_entropy(self.network(image + delta)[0], label)
        
        # AAD loss
        #loss_aad =  - torch.mean(F.cosine_similarity(self.network(image + delta)[1], sampled_py, dim=1))
        # AAD loss v2
        loss_aad =  - torch.mean(F.cosine_similarity(self.network(image + delta)[1], sampled_py, dim=1)
                                 *(self.network(image+delta)[0].max(1)[1]==label).float())
        # mart
        # kl=nn.KLDivLoss(reduction='none')
        # nat_probs = F.softmax(self.network(image)[0], dim=1)
        # adv_probs = F.softmax(self.network(image + delta)[0], dim=1)
        # tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
        # new_label = torch.where(tmp1[:, -1] == label, tmp1[:, -2], tmp1[:, -1])
        # true_probs = torch.gather(nat_probs, 1, (label.unsqueeze(1)).long()).squeeze()
        # loss_adv = F.cross_entropy(self.network(image + delta)[0], label)+ F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_label)
        # loss_robust = (1.0 / image.size(0)) * torch.sum(
        #     torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
        # loss_total = loss_adv + float(5) * loss_robust
        # + self.gamma * loss_aad

        # trades
        loss_clean = F.cross_entropy(self.network(image)[0], label)
        loss_robust = F.kl_div(F.log_softmax(self.network(image + delta)[0],dim=1),
                                                 F.softmax(self.network(image)[0],dim=1),
                                                 reduction='batchmean')
        loss_total=loss_clean+float(6) * loss_robust + self.gamma * loss_aad
        
        # loss_total = (1 - self.beta) * loss_clean + self.beta * loss_robust + self.gamma * loss_aad
        return loss_total

    def train_epoch(self):
        self.network.train()
        #prefittted vine copula model in trainset
        cln_vine=load_vinemodel('./trained_model/tinyimagenet/nt/beta_0/epoch150_vinemodel_2.pkl')
        # cln_vine=load_vinemodel('./trained_model/cifar-10/nt/beta_0/epoch150_vinemodel_v2.pkl')
        
        for batch_idx, (image, label) in enumerate(self.train_loader):
            image, label = image.to(self.device), label.to(self.device)
            self.optim.zero_grad()
            
            batch_size=image.size(0)
            fixed_noise = torch.rand(batch_size, self.args.out_dim).to(self.device)
            
            fixed_noise=fixed_noise.clamp(min=1e-6,max=1-1e-6)# 确保 fixed_noise 的元素在 (0, 1) 范围内
            sampled_r = rvinecop.inverse_rosenblatt(fixed_noise.cpu().numpy(), cln_vine)
            sampled_py = torch.Tensor(np.asarray(sampled_r)).view(batch_size, -1).to(self.device)
            
            # generate attack
            delta = pgd(self.args, self.network, image, label)
            loss_total = self.loss(image, delta, label,sampled_py)
            loss_total.backward()
            self.optim.step()

def main():
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--device", type=str, default="cuda:0,1")
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument("--dataset", type=str, default="TinyImageNet")  # "CIFAR-100", "TinyImageNet"
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--out_dim", type=int, default=50)
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=5)
    
    # Loss parameters
    parser.add_argument("--gamma", type=float, default=1)  # gamma=0 w/o AAD; gamma=1 w/ AAD
    parser.add_argument("--beta", type=float, default=1)  # beta=0 natural training; beta=1 adversarial training
    
    # Attack parameters
    parser.add_argument("--epsilon", type=float, default=0.031)
    parser.add_argument("--alpha", type=float, default=0.007)
    parser.add_argument("--num_iter", type=int, default=10)
    
    args = parser.parse_args()
    print(args)

    if not (0 <= args.beta <= 1):
        sys.exit('Wrong beta range. beta should be in [0, 1]')
    

    torch.manual_seed(args.seed)
    trainer = AT(args)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        trainer.adjust_learning_rate(epoch)
        print(f'Epoch: {epoch}, Learning Rate: {round(trainer.optim.param_groups[0]["lr"], 5)}')
        
        trainer.train_epoch()
        trainer.train_eval(epoch)
        trainer.test_eval(epoch)

    t1 = time.time()
    training_time = t1 - t0
    training_time = format_time(training_time)
    print(training_time)

    torch.save(trainer.network.state_dict(), os.path.join(trainer.path, f'epoch_{epoch}.pth'))

if __name__ == '__main__':
    main()
