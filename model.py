import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision.models import resnet

__all__ = ['ASCL', 'LinearHead']

class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


dim_dict = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
}


class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, dataset='cifar10', projection_dim=128, arch=None, bn_splits=8):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)

        net = resnet_arch(num_classes=dim_dict[arch], norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                if dataset == 'stl10' or dataset == 'tinyimagenet':
                    self.net.append(module)
                continue
            if name == 'fc':
                self.net.append(nn.Flatten(1))
                continue
            self.net.append(module)
        self.net = nn.Sequential(*self.net)

        self.feat_dim = dim_dict[arch]
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, 2048),
            nn.ReLU(True),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x, feat=False):
        x = self.net(x)
        if feat:
            return x
        else:
            x = self.projector(x)
            return x


"""### Define MoCo wrapper"""


class ASCL(nn.Module):
    def __init__(self, dataset='cifar10', model='moco', dim=128, mem_size=4096, m=0.99, T1=0.1, T2=0.05, arch='resnet18', type='ascl', nn_num=1):
        super(ASCL, self).__init__()

        self.K = K
        self.m = m
        self.T1 = T1
        self.T2 = T2
        self.type = type
        self.model = model
        self.nn_num = nn_num
        # create the encoders
        self.encoder_q = ModelBase(dataset=dataset, projection_dim=dim, arch=arch)
        self.encoder_k = ModelBase(dataset=dataset, projection_dim=dim, arch=arch)

        if self.model == 'byol':
            self.predictor = torch.nn.Sequential(torch.nn.Linear(dim, 512),
                                                 torch.nn.BatchNorm1d(512),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(512, 128))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.max_entropy = np.log(self.K)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('labels', -1 * torch.ones(self.K).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        self.labels[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def _forward_moco(self, im_q, im_k, targets):
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshufflek = self._batch_shuffle_single_gpu(im_k)
            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized
            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshufflek)

            # pseudo logits: NxK
            logits_pd = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
            logits_pd /= self.T2

            labels = torch.zeros(logits_pd.size(0), logits_pd.size(1)+1).cuda()
            if self.type == 'ascl':
                labels[:, 0] = 1.0
                pseudo_labels = F.softmax(logits_pd, 1)
                log_pseudo_labels = F.log_softmax(logits_pd, 1)
                entropy = -torch.sum(pseudo_labels * log_pseudo_labels, dim=1, keepdim=True)
                c = 1 - entropy / self.max_entropy
                pseudo_labels = self.nn_num * c * pseudo_labels  # num of neighbors * uncertainty * pseudo_labels
                pseudo_labels = torch.minimum(pseudo_labels,
                                              torch.tensor(1).to(pseudo_labels.device))  # upper thresholded by 1
                labels[:, 1:] = pseudo_labels  # summation <= c*K <= K

            elif self.type == 'ahcl':
                labels[:, 0] = 1.0
                _, nn_index = logits_pd.topk(self.nn_num, dim=1, largest=True)
                hard_labels = torch.zeros_like(logits_pd, device=logits_pd.device).scatter(1, nn_index, 1)
                pseudo_labels = F.softmax(logits_pd, 1)
                log_pseudo_labels = F.log_softmax(logits_pd, 1)
                entropy = -torch.sum(pseudo_labels * log_pseudo_labels, dim=1, keepdim=True)
                c = 1 - entropy / self.max_entropy
                labels[:, 1:] = hard_labels * c  # summation = c*K <= K

            elif self.type == 'hard':
                labels[:, 0] = 1.0
                _, nn_index = logits_pd.topk(self.nn_num, dim=1, largest=True)
                hard_labels = torch.zeros_like(logits_pd, device=logits_pd.device).scatter(1, nn_index, 1)
                labels[:, 1:] = hard_labels  # summation = K

            else:  # no extra neighbors [moco]
                labels[:, 0] = 1.0

        # label normalization
        labels = labels / labels.sum(dim=1, keepdim=True)

        # forward pass
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T1

        loss = -torch.sum(labels.detach() * F.log_softmax(logits, 1), 1).mean()

        self._dequeue_and_enqueue(k, targets)

        return loss

    def _forward_byol(self, im_q, im_k, targets):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        q = self.encoder_q(im_q)  # queries: NxC
        k = self.encoder_k(im_k)  # keys: NxC
        hq = self.predictor(q)
        hq = F.normalize(hq, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        # ascl
        if self.type == 'ascl':
            with torch.no_grad():
                distk = torch.einsum('nc,ck->nk', [k, k.T])
                logits_pd = distk[~torch.eye(distk.shape[0], dtype=bool)].reshape(distk.shape[0], -1)
                pseudo_labels = F.softmax(logits_pd / self.T2, 1)
                log_pseudo_labels = F.log_softmax(logits_pd / self.T2, 1)
                entropy = -torch.sum(pseudo_labels * log_pseudo_labels, dim=1, keepdim=True)
                max_entropy = np.log(pseudo_labels.shape[1])
                c = 1 - entropy / max_entropy
                pseudo_labels = self.nn_num * c * pseudo_labels
                pseudo_labels = torch.minimum(pseudo_labels,
                                              torch.tensor(1).to(pseudo_labels.device))  # upper thresholded by 1

                labels = torch.zeros_like(distk).cuda()
                labels.fill_diagonal_(1)
                labels[~np.eye(distk.shape[0], dtype=bool)] = pseudo_labels.reshape(-1)
                labels = labels / labels.sum(dim=1, keepdim=True)
            dist_real = torch.einsum('nc,ck->nk', [hq, k.T.detach()])

            loss = 2 - 2 * (labels.detach() * dist_real).sum(dim=-1)
        else:
            loss = 2 - 2 * (hq * k.detach()).sum(dim=-1)

        self._dequeue_and_enqueue(k, targets)

        return loss

    def forward(self, im_q, im_k, targets):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        if self.model == 'byol':
            loss = self._forward_byol(im_q, im_k, targets)
        elif self.model == 'moco':
            loss = self._forward_moco(im_q, im_k, targets)
        else:
            raise ValueError(f'Wrong model! Should be moco or byol rather than {self.model}!')
        return loss

class LinearHead(nn.Module):
    def __init__(self, net, dim_in=2048, num_class=1000):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(dim_in, num_class)

        for param in self.net.parameters():
            param.requires_grad = False

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            feat = self.net(x, feat=True)
        return self.fc(feat)
