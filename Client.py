import random
from typing import List
import copy
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms

from FederatedTask import FederatedTask
from models.extractor import FeatureExtractor
from models.model import is_bn_relavent
from losses.loss_functions import normal_loss, backdoor_loss, nc_evasion_loss, sentinet_evasion_loss, norm_loss, \
    neural_cleanse_part1, model_similarity_loss
import time
from torch.optim import lr_scheduler

from utils.min_norm_solvers import MGDASolver


def adjust_learning_rate(optimizer, factor=0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * factor


class Clientbase:

    def __init__(self, model, optimizer, device):
        self.local_model = model
        # print("device:",device)
        self.local_model.to(device)
        self.optimizer = optimizer
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[50], gamma=1.0)
        # robustlr [5,20,40,70]
        # bulyan [5,30,70]
        self.criterion = None

    def send_gradients(self):
        pass

    def send_weights(self):
        pass

    def get_parameters(self):
        # for param in self.local_model.parameters():
        #     param.detach()
        return self.local_model.parameters()

    # def set_model_weights(self, model):
    #     for local_param, global_param in zip(self.local_model.parameters(), model.parameters()):
    #         local_param.data = global_param.data.clone()

    def set_model_weights(self, model):
        model_weights = model.state_dict()
        client_weights = self.local_model.state_dict()
        for layer in model_weights.keys():
            if not is_bn_relavent(layer):
                client_weights[layer] = model_weights[layer].clone().detach()
        self.local_model.load_state_dict(client_weights)

    def freeze_model_layers(self, freezing_max_id):
        if freezing_max_id <= 0:
            return
        for i, layer in enumerate(self.local_model.named_parameters()):
            layer[1].requires_grad = False
            if i == freezing_max_id:
                return

    def regrad_model_layers(self):
        for i, layer in enumerate(self.local_model.named_parameters()):
            layer[1].requires_grad = True


class Client(Clientbase):

    def __init__(self, client_id, model, optimizer, is_malicious, dataset, local_epoch, batch_size, attacks, device):
        super().__init__(model, optimizer, device)
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.dataset = dataset
        self.n_sample = len(dataset)
        self.local_epoch = local_epoch
        self.device = device
        self.handcraft_rnd = 0
        self.train_rnd = 0

        # A benign client should not have self.attacks
        self.attacks = attacks if self.is_malicious else None

        if not self.is_malicious or not self.attacks.handcraft:
            self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        else:
            idxs = range(self.n_sample)
            nt = int(0.8 * self.n_sample)
            train_ids, test_ids = idxs[:nt], idxs[nt:]
            train_dataset = Subset(dataset, train_ids)
            handcraft_dataset = Subset(dataset, test_ids)
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                           drop_last=True)
            self.handcraft_loader = DataLoader(handcraft_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                               drop_last=True)
            self.attacks.previous_global_model = copy.deepcopy(model)

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def reset_loader(self):
        batch_size = self.handcraft_loader.batch_size
        shuffled_idxs = random.sample(range(self.n_sample), k=self.n_sample)
        nt = int(0.8 * self.n_sample)
        train_ids, test_ids = shuffled_idxs[:nt], shuffled_idxs[nt:]
        train_dataset = Subset(self.dataset, train_ids)
        handcraft_dataset = Subset(self.dataset, test_ids)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                       drop_last=True)
        self.handcraft_loader = DataLoader(handcraft_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                           drop_last=True)

    def reset_optimizer(self, params) -> optim.Optimizer:
        if params.optimizer == 'SGD':
            optimizer = optim.SGD(self.local_model.parameters(),
                                  lr=params.lr,
                                  weight_decay=params.decay,
                                  momentum=params.momentum)
        elif params.optimizer == 'Adam':
            optimizer = optim.Adam(self.local_model.parameters(), lr=params.lr, weight_decay=params.decay)
        else:
            raise ValueError(f'No optimizer:{self.optimizer}')
        return optimizer

    def compute_all_losses_and_grads(self, loss_tasks, model, criterion, batch, backdoor_batch, compute_grad=True):
        grads, loss_values = dict(), dict()
        backdoor_label = None if self.attacks is None else self.attacks.backdoor_label

        if self.attacks is not None and 'neural_cleanse' in loss_tasks:
            nc_model = self.attacks.nc_model
            nc_p_norm = self.attacks.nc_p_norm
            nc_optim = self.attacks.nc_optim
            neural_cleanse_part1(nc_model, model, batch, backdoor_batch, nc_p_norm, nc_optim)

        for task in loss_tasks:
            if compute_grad:
                model.zero_grad()
            if task == 'normal':
                loss_values[task], grads[task] = normal_loss(model, criterion, batch.inputs, batch.labels, compute_grad)
            elif task == 'backdoor':
                loss_values[task], grads[task] = backdoor_loss(model, criterion, backdoor_batch.inputs,
                                                               backdoor_batch.labels, compute_grad)
            elif task == 'stealth':
                loss_values[task], grads[task] = backdoor_loss(model, criterion, backdoor_batch.inputs,
                                                               backdoor_batch.labels, compute_grad)
            elif task == 'neural_cleanse':
                loss_values[task], grads[task] = nc_evasion_loss(self.attacks.nc_model, model, batch.inputs,
                                                                 batch.labels, compute_grad)
            elif task == 'sentinet_evasion':
                loss_values[task], grads[task] = sentinet_evasion_loss(backdoor_label, model, batch.inputs,
                                                                       backdoor_batch.inputs, backdoor_batch.labels,
                                                                       compute_grad)
            elif task == 'mask_norm':
                loss_values[task], grads[task] = norm_loss(self.attacks.nc_p_norm, model, compute_grad)
            elif task == 'neural_cleanse_part1':
                loss_values[task], grads[task] = normal_loss(model, criterion, batch.inputs, backdoor_batch.labels,
                                                             compute_grad)

        return loss_values, grads

    def compute_blind_loss(self, model, batch, does_attack=True):
        if self.attacks is None or not does_attack:
            loss_tasks = ['normal']
            loss_values, grads = self.compute_all_losses_and_grads(loss_tasks, model, self.criterion, batch, None,
                                                                   compute_grad=False)
            return torch.mean(loss_values['normal'])
        # malicious client actions
        elif self.attacks is not None and does_attack:
            # batch = batch.clip(self.attacks)
            loss_tasks = self.attacks.loss_tasks
            backdoor_batch = self.attacks.synthesizer.make_backdoor_batch(batch, attack=does_attack)
            scale = dict()

            loss_values, grads = None, None

            # if 'neural_cleanse' in loss_tasks:
            #     self.attacks.neural_cleanse_part1(model, batch, backdoor_batch)

            if len(loss_tasks) == 1:
                loss_values, grads = self.compute_all_losses_and_grads(loss_tasks, model, self.criterion, batch,
                                                                       backdoor_batch, compute_grad=False)
                scale = {loss_tasks[0]: 1.0}
            else:
                if self.attacks.loss_balance == 'MGDA':
                    loss_values, grads = self.compute_all_losses_and_grads(loss_tasks, model, self.criterion, batch,
                                                                           backdoor_batch, compute_grad=True)
                    scale = MGDASolver.get_scales(grads, loss_values, self.attacks.mgda_normalize, loss_tasks)
                elif self.attacks.loss_balance == 'fixed':
                    loss_values, grads = self.compute_all_losses_and_grads(loss_tasks, model, self.criterion, batch,
                                                                           backdoor_batch, compute_grad=False)
                    for task in loss_tasks:
                        scale[task] = self.attacks.params.fixed_scales[task]
                else:
                    raise ValueError(f'Please choose between `MGDA` and `fixed`')
            blind_loss = self.attacks.scale_losses(loss_tasks, loss_values, scale)
            return blind_loss

    def scale_updates(self, raw_weights, ratio=1):
        model_weights = self.local_model.state_dict()
        for k in model_weights.keys():
            if not is_bn_relavent(k):
                model_weights[k] = raw_weights[k] + (model_weights[k] - raw_weights[k]) * (ratio - 1)

        self.local_model.load_state_dict(model_weights)
    
    def get_grad_mask_on_cv(self, task, ratio=0.9):
        """Generate a gradient mask based on the given dataset, in the experiment we apply ratio=0.9 by default"""     
        model = self.local_model
        model.train()
        
        model.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()

        for i, data in enumerate(self.train_loader):
            batch = task.get_batch(i, data)
            outputs = model(batch.inputs)
            loss = criterion(outputs, batch.labels)
            loss.backward(retain_graph=True)
        
        mask_grad_list = []
        
        grad_list  = []
        grad_abs_sum_list = []
        k_layer = 0
        
        for _, params in model.named_parameters():
            if params.requires_grad:
                grad_list.append(params.grad.abs().view(-1))
                grad_abs_sum_list.append(params.grad.abs().view(-1).sum().item())
                k_layer += 1
        
        
        grad_list = torch.cat(grad_list).cuda()
        _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
        mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
        mask_flat_all_layer[indices] = 1.0

        count = 0
        percentage_mask_list = []
        k_layer = 0
        grad_abs_percentage_list = []
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients_length = len(parms.grad.abs().view(-1))

                mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                count += gradients_length
                percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                percentage_mask_list.append(percentage_mask1)
                grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

                k_layer += 1

        model.zero_grad()
        return mask_grad_list
    
    
    def apply_grad_mask(self, model, mask_grad_list):
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)

    
    def neurotoxin_train(self, task):
        self.train_rnd = self.train_rnd + 1
        model = self.local_model
        local_epoch = self.local_epoch
        raw_model = copy.deepcopy(model)
        model.train()
        
        mask_grad_list = self.get_grad_mask_on_cv(task)

        for epoch in range(local_epoch):
            batch_losses = list()
            normal_losses = list()
            # Record Training Time
            torch.cuda.synchronize()
            start = time.time()
            for i, data in enumerate(self.train_loader):
                batch = task.get_batch(i, data)
                self.optimizer.zero_grad()
                loss = self.compute_blind_loss(model, batch, does_attack=True)
                if self.is_malicious:
                    sim_factor = self.attacks.params.model_similarity_factor
                    loss = (1-sim_factor) * loss + sim_factor * model_similarity_loss(raw_model, model)
                
                
                loss.backward(retain_graph=True)
                self.apply_grad_mask(model, mask_grad_list)
                
                self.optimizer.step()
                batch_losses.append(loss.item())
            # Test time
            torch.cuda.synchronize()
            end = time.time()
            train_time = end - start
            print("client:{} epoch:{} mal:{} loss:{} time:{}".format(self.client_id, epoch, self.is_malicious,
                                                                     np.mean(batch_losses), round(train_time, 2)))
        self.scheduler.step()


    def train(self, task):
        if self.is_malicious and self.attacks.neurotoxin:
            print("use neurotoxin-train as normal-train")
            self.neurotoxin_train(task)
            return
            
        self.train_rnd = self.train_rnd + 1
        model = self.local_model
        local_epoch = self.local_epoch
        raw_model = copy.deepcopy(model)
        model.train()

        # if self.is_malicious:
        #     if not self.attacks.handcraft:
        #         local_epoch = 5
        
        for epoch in range(local_epoch):
            batch_losses = list()
            normal_losses = list()
            # Record Training Time
            torch.cuda.synchronize()
            start = time.time()
            for i, data in enumerate(self.train_loader):
                batch = task.get_batch(i, data)
                self.optimizer.zero_grad()
                loss = self.compute_blind_loss(model, batch, does_attack=True)
                if self.is_malicious:
                    sim_factor = self.attacks.params.model_similarity_factor
                    loss = (1-sim_factor) * loss + sim_factor * model_similarity_loss(raw_model, model)
                loss.backward()

                self.optimizer.step()
                batch_losses.append(loss.item())
            # Test time
            torch.cuda.synchronize()
            end = time.time()
            train_time = end - start
            print("client:{} epoch:{} mal:{} loss:{} time:{}".format(self.client_id, epoch, self.is_malicious,
                                                                     np.mean(batch_losses), round(train_time, 2)))
        self.scheduler.step()

    def handcraft(self, task):
        self.handcraft_rnd = self.handcraft_rnd + 1
        if self.is_malicious and self.attacks.handcraft:
            model = self.local_model
            model.eval()
            handcraft_loader, train_loader = self.handcraft_loader, self.train_loader

            if self.attacks.previous_global_model is None:
                self.attacks.previous_global_model = copy.deepcopy(model)
                return
            candidate_weights = self.attacks.search_candidate_weights(model, proportion=0.1)
            self.attacks.previous_global_model = copy.deepcopy(model)

            if self.attacks.params.handcraft_trigger:
                print("Optimize Trigger:")
                self.attacks.optimize_backdoor_trigger(model, candidate_weights, task, handcraft_loader)

            print("Inject Candidate Filters:")
            diff = self.attacks.inject_handcrafted_filters(model, candidate_weights, task, handcraft_loader)
            if diff is not None and self.handcraft_rnd % 3 == 1:
                print("Rnd {}: Inject Backdoor FC".format(self.handcraft_rnd))
                self.attacks.inject_handcrafted_neurons(model, candidate_weights, task, diff, handcraft_loader)

    def get_conv_rank(self, task: FederatedTask, n_test_batch, location):
        assert (location in ['last'])
        train_loader = self.train_loader
        self.local_model.eval()
        final_activations = None
        for i, data in enumerate(train_loader):
            batch = task.get_batch(i, data)
            final_activation = self.local_model.final_activations(batch.inputs)
            if final_activations is None:
                final_activations = torch.zeros_like(final_activation) + final_activation
            else:
                final_activations = final_activations + final_activation

            if i + 1 == n_test_batch:
                break
                
        final_activations = final_activations / min(n_test_batch, len(train_loader))
        channel_activations = torch.sum(final_activations, dim=[0, 2, 3])
        _, channel_idxs = torch.sort(channel_activations, descending=False)
        _, channel_ranks = torch.sort(channel_idxs)
        return channel_ranks

    def idle(self):
        self.train_rnd = self.train_rnd + 1
        self.handcraft_rnd = self.handcraft_rnd + 1
        self.scheduler.step()
