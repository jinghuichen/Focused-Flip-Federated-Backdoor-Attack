import copy
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from torch import optim, nn, kl_div
from typing import List

from torch.utils.data import DataLoader

from Params import Params
from Client import Client
from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from torch.nn import Module
from Attacks import Attacks, get_conv_weight_names, get_accuracy
# Type Definition
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import math
from sklearn.cluster import DBSCAN
# define all the operations on global models
from models.resnet import resnet18, ResNet
from FederatedTask import Cifar10FederatedTask, TinyImagenetFederatedTask

client_group = List[Client]


class Serverbase:
    def __init__(self, model, optimizer, device):
        self.global_model = model
        # print("server:",device)
        self.global_model.to(device)
        self.optimizer = optimizer

    def aggregate_gradients(self):
        pass

    def aggregate_weights(self):
        pass

    # def add_weights(self, client, ratio):
    #     #print(ratio)
    #     for server_param, user_param in zip(self.global_model.parameters(), client.get_parameters()):
    #         server_param.data = server_param.data + user_param.data.clone() * ratio

    def add_weights(self, averaged_weights: OrderedDict, client_weights: OrderedDict, ratio):
        for layer in client_weights.keys():
            averaged_weights[layer] = averaged_weights[layer] + client_weights[layer] * ratio

    def robust_lr_add_weights(self, original_params, robust_lrs, update, prop):
        for layer in original_params.keys():
            if 'running' in layer or 'tracked' in layer:
                original_params[layer] = original_params[layer] + update[layer] * prop
            else:
                original_params[layer] = original_params[layer] + update[layer] * prop * robust_lrs[layer]

    def get_global_model_norm(self):
        squared_sum = 0
        for name, layer in self.global_model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)


class ServerAvg(Serverbase):
    def __init__(self, model, optimizer, n_clients, chosen_rate, dataset, batch_size, device):
        super().__init__(model, optimizer, device)
        self.n_clients = n_clients
        self.chosen_rate = chosen_rate

        if dataset is not None:
            self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.device = device

    def select_participated_clients(self, fixed_mal):
        n_chosen = int(self.chosen_rate * self.n_clients)
        candidate_ids = list()
        for c in range(self.n_clients):
            if not c in fixed_mal:
                candidate_ids.append(c)
        selected_benigns = np.random.choice(candidate_ids, n_chosen - len(fixed_mal), replace=False)
        return list(selected_benigns) + list(fixed_mal)

    def broadcast_model_weights(self, clients: client_group):
        self.global_model.train()
        for client in clients:
            client.set_model_weights(self.global_model)

    def broadcast_global_optimizer(self, clients: client_group):
        pass

    #     def compute_robustLR(self, updates):
    #         layers = updates[0].keys()
    #         signed_weights = OrderedDict()
    #         robust_lrs = OrderedDict()
    #         for layer, weight in self.global_model.state_dict().items():
    #             signed_weights[layer] = torch.zeros_like(weight)
    #             robust_lrs[layer] = torch.zeros_like(weight)

    #         for layer in layers:
    #             for update in updates:
    #                 signed_weights[layer] += torch.sign(update[layer])
    #             signed_weights[layer] = torch.abs(signed_weights[layer])
    #             robust_lrs[layer][signed_weights[layer] >= 2] = 1.0
    #             robust_lrs[layer][signed_weights[layer] <  2] = -1.0

    #         return robust_lrs

    def compute_robustLR(self, updates):
        layers = updates[0].keys()
        # signed_weights = OrderedDict()
        robust_lrs = OrderedDict()
        for layer, weight in self.global_model.state_dict().items():
            # signed_weights[layer] = torch.zeros_like(weight)
            robust_lrs[layer] = torch.zeros_like(weight)

        for layer in layers:
            for update in updates:
                robust_lrs[layer] += torch.sign(update[layer])
            robust_lrs[layer] = torch.abs(robust_lrs[layer])
            robust_lrs[layer][robust_lrs[layer] >= 2] = 1.0
            robust_lrs[layer][robust_lrs[layer] != 1.0] = -1.0
        return robust_lrs

    def aggregate_global_model(self, clients: client_group, chosen_ids, pts):
        assert not clients is None and len(clients) > 0
        averaged_weights = OrderedDict()
        for layer, weight in self.global_model.state_dict().items():
            averaged_weights[layer] = torch.zeros_like(weight)

        total_prop = 0
        if pts is None:
            pts = [1 for i in range(len(chosen_ids))]

        for pt, id in zip(pts, chosen_ids):
            client = clients[id]
            total_prop = total_prop + client.n_sample * pt

        for pt, id in zip(pts, chosen_ids):
            client = clients[id]
            prop = client.n_sample * pt / total_prop
            self.add_weights(averaged_weights, client.local_model.state_dict(), prop)

        # for client in clients:
        #     client.local_model.to('cpu')

        self.global_model.load_state_dict(averaged_weights)

    def collect_conv_ranks(self, task, clients: client_group, chosen_ids, pts):
        client_ranks = list()
        for id in chosen_ids:
            client = clients[id]
            client_ranks.append(client.get_conv_rank(task, n_test_batch=10, location='last'))

        averaged_client_rank = torch.zeros_like(client_ranks[0])
        for client_rank in client_ranks:
            averaged_client_rank = averaged_client_rank + client_rank
        averaged_client_rank = averaged_client_rank / len(client_ranks)
        _, prune_orders = torch.sort(averaged_client_rank, descending=True)
        prune_orders = prune_orders.cpu().numpy().tolist()
        print("prune_orders:", prune_orders)
        return prune_orders

    def conv_pruning(self, task, orders):
        self.global_model.eval()
        model_weights = self.global_model.state_dict()
        final_conv = get_conv_weight_names(self.global_model)[-1]

        final_gamma, final_bias = None, None
        if isinstance(self.global_model, ResNet):
            final_gamma = final_conv.replace("conv", "bn")
            final_bias = final_conv.replace("conv", "bn").replace('weight', 'bias')

        last_accuracy = get_accuracy(self.global_model, task, self.train_loader)
        for i, conv_id in enumerate(orders):
            original_weights = self.global_model.state_dict()

            model_weights[final_conv][conv_id] = torch.zeros_like(model_weights[final_conv][conv_id])
            if final_bias is not None and final_gamma is not None:
                model_weights[final_gamma][conv_id] = 0.0
                model_weights[final_bias][conv_id] = 0.0

            self.global_model.load_state_dict(model_weights)
            current_accuracy = get_accuracy(self.global_model, task, self.train_loader)

            if last_accuracy - current_accuracy >= 0.01:
                self.global_model.load_state_dict(original_weights)
                print("prune:{}".format(i))
                return

    def adjust_extreme_parameters(self, threshold):
        self.global_model.eval()
        model_weights = self.global_model.state_dict()
        final_conv = get_conv_weight_names(self.global_model)[-1]
        min_w = float(torch.mean(model_weights[final_conv]) - threshold * torch.std(model_weights[final_conv]))
        max_w = float(torch.mean(model_weights[final_conv]) + threshold * torch.std(model_weights[final_conv]))
        model_weights[final_conv][model_weights[final_conv] > max_w]= 0.0
        model_weights[final_conv][model_weights[final_conv] < min_w]= 0.0
        p_zero = torch.sum((model_weights[final_conv] == 0.0).int()).item() / model_weights[final_conv].numel()
        print("Adjust Extreme Value: {}".format(p_zero))
        self.global_model.load_state_dict(model_weights)

    def sign_voting_aggregate_global_model(self, clients: client_group, chosen_ids, pts):
        assert not clients is None and len(clients) > 0
        original_params = self.global_model.state_dict()

        total_sample = 0
        for id in chosen_ids:
            client = clients[id]
            total_sample = total_sample + client.n_sample

        # collect client updates
        updates = list()
        for id in chosen_ids:
            client = clients[id]
            local_params = client.local_model.state_dict()
            update = OrderedDict()
            for layer, weight in local_params.items():
                update[layer] = local_params[layer] - original_params[layer]
            updates.append(update)

        # compute_total_update
        robust_lrs = self.compute_robustLR(updates)
        # count signs：
        flip_analysis = dict()
        for layer in robust_lrs.keys():
            n_flip = torch.sum(torch.gt(robust_lrs[layer], 0.0).int())
            n_unflip = torch.sum(torch.lt(robust_lrs[layer], 0.0).int())
            flip_analysis[layer] = [n_flip, n_unflip]

        for i, id in enumerate(chosen_ids):
            client = clients[id]
            prop = client.n_sample / total_sample
            self.robust_lr_add_weights(original_params, robust_lrs, updates[i], prop)

        self.global_model.load_state_dict(original_params)
        return flip_analysis

    def compute_pairwise_distance(self, updates):
        def pairwise(u1, u2):
            ks = u1.keys()
            dist = 0
            for k in ks:
                if 'tracked' in k:
                    continue
                d = u1[k] - u2[k]
                dist = dist + torch.sum(d * d)
            return round(float(torch.sqrt(dist)), 2)

        scores = [0 for u in range(len(updates))]
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                dist = pairwise(updates[i], updates[j])
                scores[i] = scores[i] + dist
                scores[j] = scores[j] + dist
        return scores

    def bulyan_aggregate_global_model(self, clients: client_group, chosen_ids, pts):
        assert not clients is None and len(clients) > 0
        n_mal = 4
        original_params = self.global_model.state_dict()

        # collect client updates
        updates = list()
        for id in chosen_ids:
            client = clients[id]
            local_params = client.local_model.state_dict()
            update = OrderedDict()
            for layer, weight in local_params.items():
                update[layer] = local_params[layer] - original_params[layer]
            updates.append(update)

        temp_ids = list(copy.deepcopy(chosen_ids))

        krum_updates = list()
        n_ex = 2 * n_mal
        # print("Bulyan Stage 1：", len(updates))
        for i in range(len(chosen_ids)-n_ex):
            scores = self.compute_pairwise_distance(updates)
            n_update = len(updates)
            threshold = sorted(scores)[0]
            for k in range(n_update - 1, -1, -1):
                if scores[k] == threshold:
                    print("client {} is chosen:".format(temp_ids[k], round(scores[k], 2)))
                    krum_updates.append(updates[k])
                    del updates[k]
                    del temp_ids[k]
                    
        # print("Bulyan Stage 2：", len(krum_updates))    
        bulyan_update = OrderedDict()
        layers = krum_updates[0].keys()
        for layer in layers:
            bulyan_layer = None
            for update in krum_updates:
                bulyan_layer = update[layer][None, ...] if bulyan_layer is None else torch.cat(
                    (bulyan_layer, update[layer][None, ...]), 0)

            med, _ = torch.median(bulyan_layer, 0)
            _, idxs = torch.sort(torch.abs(bulyan_layer - med), 0)
            bulyan_layer = torch.gather(bulyan_layer, 0, idxs[:-n_ex, ...])
            # print("bulyan_layer",bulyan_layer.size())
            # bulyan_update[layer] = torch.mean(bulyan_layer, 0)
            # print(bulyan_layer)
            if not 'tracked' in layer:
                bulyan_update[layer] = torch.mean(bulyan_layer, 0)
            else:
                bulyan_update[layer] = torch.mean(bulyan_layer*1.0, 0).long()
            original_params[layer] = original_params[layer] + bulyan_update[layer]

        self.global_model.load_state_dict(original_params)
    
    def deepsight_aggregate_global_model(self, clients: client_group, chosen_ids, task, pts):
        def ensemble_cluster(neups, ddifs, biases):
            biases = np.array([bias.cpu().numpy() for bias in biases])
            #neups = np.array([neup.cpu().numpy() for neup in neups])
            #ddifs = np.array([ddif.cpu().detach().numpy() for ddif in ddifs])
            N = len(neups)
            # use bias to conduct DBSCAM
            # biases= np.array(biases)
            cosine_labels = DBSCAN(min_samples=3,metric='cosine').fit(biases).labels_
            print("cosine_cluster:{}".format(cosine_labels))
            # neups=np.array(neups)
            neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
            print("neup_cluster:{}".format(neup_labels))
            ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
            print("ddif_cluster:{}".format(ddif_labels))

            dists_from_cluster = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                        neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j]))/3.0
                    dists_from_cluster[j, i] = dists_from_cluster[i, j]
                    
            print("dists_from_clusters:")
            print(dists_from_cluster)
            ensembled_labels = DBSCAN(min_samples=3,metric='precomputed').fit(dists_from_cluster).labels_

            return ensembled_labels
        
        global_weight = list(self.global_model.state_dict().values())[-2]
        global_bias = list(self.global_model.state_dict().values())[-1]

        biases = [(list(clients[i].local_model.state_dict().values())[-1] - global_bias) for i in chosen_ids]
        weights = [list(clients[i].local_model.state_dict().values())[-2] for i in chosen_ids]

        n_client = len(chosen_ids)
        cosine_similarity_dists = np.array((n_client, n_client))
        neups = list()
        n_exceeds = list()

        # calculate neups
        sC_nn2 = 0
        for i in range(len(chosen_ids)):
            C_nn = torch.sum(weights[i]-global_weight, dim=[1]) + biases[i]-global_bias
            # print("C_nn:",C_nn)
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2
            
            C_max = torch.max(C_nn2).item()
            threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)
        # normalize
        neups = np.array([(neup/sC_nn2).cpu().numpy() for neup in neups])
        print("n_exceeds:{}".format(n_exceeds))
        rand_input = None
        if isinstance(task, Cifar10FederatedTask):
            # 256 can be replaced with smaller value
            rand_input = torch.randn((256, 3, 32, 32)).to(self.device)
        elif isinstance(task, TinyImagenetFederatedTask):
            # 256 can be replaced with smaller value
            rand_input = torch.randn((256, 3, 64, 64)).to(self.device)

        global_ddif = torch.mean(torch.softmax(self.global_model(rand_input), dim=1), dim=0)
        # print("global_ddif:{} {}".format(global_ddif.size(),global_ddif))
        client_ddifs = [torch.mean(torch.softmax(clients[i].local_model(rand_input), dim=1), dim=0)/ global_ddif
                        for i in chosen_ids]
        client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])
        # print("client_ddifs:{}".format(client_ddifs[0]))

        # use n_exceed to label
        classification_boundary = np.median(np.array(n_exceeds)) / 2
        
        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
        print("identified_mals:{}".format(identified_mals))
        clusters = ensemble_cluster(neups, client_ddifs, biases)
        print("ensemble clusters:{}".format(clusters))
        cluster_ids = np.unique(clusters)

        deleted_cluster_ids = list()
        for cluster_id in cluster_ids:
            n_mal = 0
            cluster_size = np.sum(cluster_id == clusters)
            for identified_mal, cluster in zip(identified_mals, clusters):
                if cluster == cluster_id and identified_mal:
                    n_mal += 1
            print("cluser size:{} n_mal:{}".format(cluster_size,n_mal))        
            if (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)
        # print("deleted_clusters:",deleted_cluster_ids)
        temp_chosen_ids = copy.deepcopy(chosen_ids)
        for i in range(len(chosen_ids)-1, -1, -1):
            # print("cluster tag:",clusters[i])
            if clusters[i] in deleted_cluster_ids:
                del chosen_ids[i]

        print("final clients length:{}".format(len(chosen_ids)))
        if len(chosen_ids)==0:
            chosen_ids = temp_chosen_ids
        self.aggregate_global_model(clients, chosen_ids, None)
        
    def clip_weight_norm(self, clip=14):
        total_norm = self.get_global_model_norm()
        print("total_norm: " + str(total_norm) + "clip: " + str(clip))
        max_norm = clip
        clip_coef = max_norm / (total_norm + 1e-6)
        current_norm = total_norm
        if total_norm > max_norm:
            for name, layer in self.global_model.named_parameters():
                layer.data.mul_(clip_coef)
            current_norm = self.get_global_model_norm()
        return current_norm

    def add_differential_privacy_noise(self, sigma=0.001, cp=False):
        if not cp:
            for name, param in self.global_model.state_dict().items():
                if 'tracked' in name or 'running' in name:
                    continue
                # print(name)
                dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                param.add_(dp_noise)
        else:
            smoothed_model = copy.deepcopy(self.global_model)
            for name, param in smoothed_model.state_dict().items():
                if 'tracked' in name or 'running' in name:
                    continue
                dp_noise = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
                param.add_(dp_noise)
            return smoothed_model

    def get_median_scores(self, task, clients: client_group, chosen_ids):
        self.global_model.train()
        median_counts = [0 for i in range(len(chosen_ids))]
        for i, data in enumerate(self.train_loader):
            batch = task.get_batch(i, data)
            median_count = task.get_median_counts(batch, clients, chosen_ids)
            median_counts = [x + y for x, y in zip(median_counts, median_count)]
        total_counts = sum(median_counts)
        normalized_median_counts = [(med_count / total_counts) for med_count in median_counts]
        return normalized_median_counts

    def get_avg_logits(self):
        pass
        # for i, data in enumerate(self.train_loader):
        #     # clear tddm
        #     batch = task.get_batch(i, data)

    def ensemble_distillation(self, task, clients: client_group, chosen_ids):
        self.global_model.train()
        for i, data in enumerate(self.train_loader):
            batch = task.get_batch(i, data)
            batch = task.get_avg_logits(batch, clients, chosen_ids)
            self.optimizer.zero_grad()
            predicted_labels = self.global_model(batch.inputs)
            kl_div_loss = nn.KLDivLoss(reduction='batchmean')(predicted_labels.softmax(dim=-1).log(),
                                                              batch.labels.softmax(dim=-1))
            kl_div_loss.backward()
            self.optimizer.step()

    def adaptive_distillation(self, task, clients: client_group, chosen_ids):
        self.global_model.train()
        for i, data in enumerate(self.train_loader):
            batch = task.get_batch(i, data)
            batch = task.get_median_logits(batch, clients, chosen_ids)
            self.optimizer.zero_grad()
            predicted_labels = self.global_model(batch.inputs)
            kl_div_loss = nn.KLDivLoss(reduction='batchmean')(predicted_labels.softmax(dim=-1).log(),
                                                              batch.labels.softmax(dim=-1))
            kl_div_loss.backward()
            self.optimizer.step()

    def fine_tuning(self, task, clients: client_group, chosen_ids):
        self.global_model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        for i, data in enumerate(self.train_loader):
            batch = task.get_batch(i, data)
            self.optimizer.zero_grad()
            predicted_labels = self.global_model(batch.inputs)
            # print("predicted_labels:",predicted_labels,batch.labels)
            loss = criterion(predicted_labels, batch.labels).mean()
            loss.backward()
            self.optimizer.step()
