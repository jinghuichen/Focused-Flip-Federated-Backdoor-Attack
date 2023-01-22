from collections import defaultdict
from typing import List

import random

import torch
import torchvision
import numpy as np
import yaml
from PIL import Image
from torch import optim, nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from Params import Params
from metrics.accuracy_metric import AccuracyMetric
from metrics.test_loss_metric import TestLossMetric
from models.resnet import resnet18
from Batch import Batch
from metrics.metric import Metric
from models.simple import SimpleNet
import os
import sys


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {
        }
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {
            i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {
            classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(val_image_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {
        }
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {
            classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {
            i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


class FederatedTask:
    params: Params = None

    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    classes = None

    model: Module = None
    optimizer: optim.Optimizer = None
    criterion: Module = None
    # scheduler: MultiStepLR = None
    metrics: List[Metric] = None

    def __init__(self, params: Params):
        self.params = params
        self.model: Module = None
        self.optimizer: optim.Optimizer = None

    def init_federated_task(self):
        self.load_data()
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.resume_model()
        self.model = self.model.to(self.params.device)

        self.criterion = self.build_criterion()
        self.metrics = [AccuracyMetric(), TestLossMetric(self.criterion)]
        self.set_input_shape()

    def load_data(self) -> None:
        raise NotImplemented

    def build_model(self) -> Module:
        raise NotImplemented

    def build_criterion(self) -> Module:
        return nn.CrossEntropyLoss(reduction='none')

    def accumulate_metrics(self, outputs=None, labels=None, specified_metrics=None):
        if specified_metrics is None:
            for metric in self.metrics:
                metric.accumulate_on_batch(outputs, labels)
        else:
            for metric in self.metrics:
                if metric.__class__.__name__ in specified_metrics:
                    metric.accumulate_on_batch(outputs, labels)

    # def report_metrics(self, epoch, prefix, tb_write, tb_prefix):
    #     return None

    def resume_model(self):
        if self.params.resume_model:
            path = "saved_models/{}".format(str(self.params.resume_model))
            loaded_params = torch.load(path, map_location=torch.device('cpu'))
            self.model.load_state_dict(loaded_params['state_dict'])
            self.params.start_epoch = loaded_params['epoch']
            self.params.lr = loaded_params.get('lr', self.params.lr)

            print(f"Loaded parameters from saved model: LR is"
                  f" {self.params.lr} and current epoch is"
                  f" {self.params.start_epoch}")

    def build_optimizer(self, model=None) -> optim.Optimizer:
        if model is None:
            model = self.model
        if self.params.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda layer: layer.requires_grad, model.parameters()),
                                  lr=self.params.lr,
                                  weight_decay=self.params.decay,
                                  momentum=self.params.momentum)
            print("optimizer:SGD")
        elif self.params.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda layer: layer.requires_grad, model.parameters()),
                                   lr=self.params.lr,
                                   weight_decay=self.params.decay)
            print("optimizer:Adam")
        else:
            raise ValueError(f'No optimizer:{self.optimizer}')
        return optimizer

    def set_input_shape(self):
        inp = self.train_dataset[0][0]
        self.params.input_shape = inp.shape

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_metric()

    def get_batch(self, batch_id, data) -> Batch:
        """Process data into a batch.

        Specific for different datasets and data loaders this method unifies the output by returning the object of class Batch.
        :param batch_id: id of the batch
        :param data: object returned by the Loader.
        :return:
        """
        inputs, labels = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(self.params.device)

    def get_avg_logits(self, batch: Batch, clients, chosen_ids) -> Batch:
        ensembled_batch = batch.clone()
        with torch.no_grad():
            total_logits = None
            for id in chosen_ids:
                client = clients[id]
                client.local_model.eval()
                logit = client.local_model(batch.inputs)
                total_logits = logit if total_logits is None else total_logits + logit
            avg_logit = total_logits / len(chosen_ids)
            ensembled_batch.labels = avg_logit
        return ensembled_batch

    def get_median_logits(self, batch: Batch, clients, chosen_ids) -> Batch:
        ensembled_batch = batch.clone()
        with torch.no_grad():
            all_logits = None
            for i, id in enumerate(chosen_ids):
                client = clients[id]
                client.local_model.eval()
                logit = client.local_model(batch.inputs)
                all_logits = logit[None, ...] if all_logits is None else torch.cat((all_logits, logit[None, ...]),
                                                                                   dim=0)
            median_logit, _ = torch.median(all_logits, dim=0)

            ensembled_batch.labels = median_logit
        return ensembled_batch

    def get_median_counts(self, batch: Batch, clients, chosen_ids) -> list:
        indice_counts = list()
        with torch.no_grad():
            all_logits = None
            for id in chosen_ids:
                client = clients[id]
                client.local_model.eval()
                logit = client.local_model(batch.inputs)
                all_logits = logit[None, ...] if all_logits is None else torch.cat((all_logits, logit[None, ...]),
                                                                                   dim=0)
            median_logit, indices = torch.median(all_logits, dim=0)
            indices = indices.view(-1).tolist()

            for i in range(len(chosen_ids)):
                indice_counts.append(indices.count(i))
        return indice_counts
    
    def sample_dirichlet_train_data(self, n_client):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indices dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as
            parameters for
            dirichlet distribution to sample number of images in each class.
        """
        alpha = self.params.heterogenuity

        total_classes = dict()
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in total_classes:
                total_classes[label].append(ind)
            else:
                total_classes[label] = [ind]

        class_size = len(total_classes[0])
        per_client_list = defaultdict(list)
        n_class = len(total_classes.keys())

        np.random.seed(111)
        for n in range(n_class):
            random.shuffle(total_classes[n])
            n_party = n_client
            if self.params.server_dataset:
                sampled_probabilities = class_size * np.random.dirichlet(np.array(n_client * [alpha] + [0.4]))
                n_party = n_party + 1
            else:
                sampled_probabilities = class_size * np.random.dirichlet(np.array(n_client * [alpha]))
            for p in range(n_party):
                n_image = int(round(sampled_probabilities[p]))
                sampled_list = total_classes[n][:min(len(total_classes[n]), n_image)]

                per_client_list[p].extend(sampled_list)
                # decrease the chosen samples
                total_classes[n] = total_classes[n][min(len(total_classes[n]), n_image):]

        # is a list to contain img_id
        return per_client_list



class TinyImagenetFederatedTask(FederatedTask):
    def __init__(self, params: Params):
        super(TinyImagenetFederatedTask, self).__init__(params)
        self.means = (0.485, 0.456, 0.406)
        self.lvars = (0.229, 0.224, 0.225)
        self.normalize = transforms.Normalize(self.means, self.lvars)
        self.data_dir = './tiny-imagenet-200/'

    def load_imagenet_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        self.train_dataset = TinyImageNet(self.data_dir, train=True, transform=transform_train)
        self.test_dataset = TinyImageNet(self.data_dir, train=False, transform=transform_test)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True,
                                       num_workers=0)

        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = [i for i in range(200)]

    def load_data(self) -> None:
        self.load_imagenet_data()

    # need to change the size of input and output
    def build_model(self) -> Module:
        if self.params.model == 'resnet18':
            if self.params.pretrained:
                model = resnet18(pretrained=True)
                model.fc = nn.Linear(512, len(self.classes))
            else:
                model = resnet18(pretrained=False, num_classes=len(self.classes))
                print("build resnet18")
            return model
        elif self.params.model == 'simple':
            if self.params.pretrained:
                raise NotImplemented
            else:
                model = SimpleNet(num_classes=len(self.classes))
            return model


class Cifar10FederatedTask(FederatedTask):
    def __init__(self, params: Params):
        super(Cifar10FederatedTask, self).__init__(params)
        self.means = (0.4914, 0.4822, 0.4465)
        self.lvars = (0.2023, 0.1994, 0.2010)
        
        self.normalize = transforms.Normalize(self.means, self.lvars)

    def load_cifar_data(self):
        if self.params.transform_train:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.params.batch_size,
                                       shuffle=True,
                                       num_workers=0)

        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size,
                                      shuffle=False, num_workers=0)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return True

    def load_data(self) -> None:
        self.load_cifar_data()

    def build_model(self) -> Module:
        if self.params.model == 'resnet18':
            if self.params.pretrained:
                model = resnet18(pretrained=True)
                # model is pretrained on ImageNet changing classes to CIFAR
                model.fc = nn.Linear(512, len(self.classes))
            else:
                model = resnet18(pretrained=False, num_classes=len(self.classes))
                print("resnet18")
            return model
        elif self.params.model == 'simple':
            if self.params.pretrained:
                raise NotImplemented
            else:
                model = SimpleNet(num_classes=len(self.classes))
            return model

if __name__ == "__main__":
    with open('configs/cifar_fed.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params = Params(**params)
    task = TinyImagenetFederatedTask(params)
    task.init_federated_task()