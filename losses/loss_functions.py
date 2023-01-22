import time

import torch
from torch import nn, autograd
from torch.nn import functional as F

from models.extractor import FeatureExtractor
from models.model import Model
from Params import Params
from utils.utils import th, record_time
from torchvision import transforms


# def compute_all_losses_and_grads(loss_tasks, attack, model, criterion, batch, batch_back, compute_grad=None):
#     grads = {}
#     loss_values = {}
#     for t in loss_tasks:
#         # if compute_grad:
#         #     model.zero_grad()
#         if t == 'normal':
#             loss_values[t], grads[t] = normal_loss(model, criterion, batch.inputs, batch.labels, grads=compute_grad)
#         elif t == 'backdoor':
#             loss_values[t], grads[t] = backdoor_loss(attack.params,
#                                                              model,
#                                                              criterion,
#                                                              batch_back.inputs,
#                                                              batch_back.labels,
#                                                              grads=compute_grad)
#         elif t == 'neural_cleanse':
#             loss_values[t], grads[t] = compute_nc_evasion_loss(
#                 attack.params, attack.nc_model,
#                 model,
#                 batch.inputs,
#                 batch.labels,
#                 grads=compute_grad)
#         elif t == 'sentinet_evasion':
#             loss_values[t], grads[t] = compute_sentinet_evasion(
#                 attack.params,
#                 model,
#                 batch.inputs,
#                 batch_back.inputs,
#                 batch_back.labels,
#                 grads=compute_grad)
#         elif t == 'mask_norm':
#             loss_values[t], grads[t] = norm_loss(attack.params, attack.nc_model,
#                                                  grads=compute_grad)
#         elif t == 'neural_cleanse_part1':
#             loss_values[t], grads[t] = compute_normal_loss(attack.params,
#                                                            model,
#                                                            criterion,
#                                                            batch.inputs,
#                                                            batch_back.labels,
#                                                            grads=compute_grad,
#                                                            )
#
#         # if loss_values[t].mean().item() == 0.0:
#         #     loss_values.pop(t)
#         #     grads.pop(t)
#         #     loss_tasks.remove(t)
#     return loss_values, grads


def normal_loss(model, criterion, inputs, labels, grads):
    outputs = model(inputs)
    loss = criterion(outputs, labels).mean()

    # if not params.dp:
    #     loss = loss.mean()

    if grads:
        grads = list(torch.autograd.grad(loss.mean(), [x for x in model.parameters() if x.requires_grad],
                                         retain_graph=True))

    return loss, grads


def compute_attention_loss(params, model, criterion, inputs, labels, grads):
    # outputs = model(inputs)
    # loss = criterion(outputs, labels).mean()

    pass


def nc_evasion_loss(nc_model: Model, model: Model, inputs, labels, grads):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    nc_model.switch_grads(False)
    outputs = model(nc_model(inputs))
    loss = criterion(outputs, labels).mean()

    if grads:
        grads = get_grads(model, loss)

    return loss, grads

def model_similarity_loss(global_model:Model, local_model:Model):
    global_model.switch_grads(False)
    global_weights=global_model.state_dict()
    local_weights=local_model.state_dict()
    layers = global_weights.keys()
    loss = 0
    for layer in layers:
        if 'tracked' in layer or 'running' in layer:
            continue
        layer_dist = global_weights[layer]-local_weights[layer]
        loss = loss + torch.sum(layer_dist*layer_dist)
    return loss


def backdoor_loss(model, criterion, backdoor_inputs, backdoor_labels, grads):
    outputs = model(backdoor_inputs)
    loss = criterion(outputs, backdoor_labels).mean()

    if grads:
        grads = get_grads(model, loss)

    return loss, grads

# Pei's idea2, as the paper
def trigger_loss3(model, criterion, inputs, backdoor_inputs, backdoor_labels, pattern, extractor, grads):
    clean_outputs = model(inputs)
    clean_activations = extractor._extracted_activations['relu'].clone().detach().mean([0, 1])
    extractor.clear_activations()

    backdoor_outputs = model(backdoor_inputs)
    backdoor_activations = extractor._extracted_activations['relu'].mean([0, 1])

    activations = backdoor_activations - clean_activations
    rewards = torch.sum(activations * activations)

    if grads:
        grads = torch.autograd.grad(rewards, pattern, retain_graph=True)

    return rewards, grads


# Pei's idea3, based on the guess on the models
def trigger_loss4(model, criterion, inputs, backdoor_inputs, backdoor_labels, pattern, extractor, device, grads):
    convs = model.state_dict()['conv1.weight']
    avg_convs = convs.mean([0])
    w = convs[0, ...].size()[1]
    assert w == 7
    resize = transforms.Resize((w, w))

    backdoor_conv_weight = resize(pattern)
    lmin, lmax = backdoor_conv_weight.min(), backdoor_conv_weight.max()
    cmin, cmax = convs.min(), convs.max()

    backdoor_conv_weight = (backdoor_conv_weight - lmin) / (lmax - lmin) * (cmax - cmin) + cmin
    backdoor_conv = nn.Conv2d(3, 1, kernel_size=7, stride=2, padding=3, bias=False).to(device)
    # print('weight:', backdoor_conv.weight.size())
    backdoor_conv.weight = torch.nn.Parameter(backdoor_conv_weight.unsqueeze(0))
    backdoor_norm = nn.BatchNorm2d(1).to(device)
    backdoor_relu = nn.ReLU(inplace=True).to(device)

    backdoor_activations = backdoor_relu(backdoor_norm(backdoor_conv(backdoor_inputs))).mean([0, 1])

    clean_outputs = model(backdoor_inputs)
    clean_activations = extractor._extracted_activations['relu'].clone().detach().mean([0, 1])

    activations = backdoor_activations - clean_activations
    euclid_dist = backdoor_conv_weight - avg_convs
    rewards = torch.sum(activations * activations) - 0.002 * torch.sum(euclid_dist * euclid_dist)

    if grads:
        grads = torch.autograd.grad(rewards, pattern, retain_graph=True)

    return rewards, grads


def attention_loss(raw_model, handcrafted_model, backdoor_inputs, pattern, grads):
    raw_model.train()
    raw_extractor = FeatureExtractor(raw_model)
    raw_extractor.insert_activation_hook(raw_model)

    handcrafted_extractor = FeatureExtractor(handcrafted_model)
    handcrafted_extractor.insert_activation_hook(handcrafted_model)

    handcrafted_outputs = handcrafted_model(backdoor_inputs)
    handcrafted_activations = handcrafted_extractor._extracted_activations['relu'].mean([0, 1])

    raw_outputs = raw_model(backdoor_inputs)
    raw_activations = raw_extractor._extracted_activations['relu'].clone().detach().mean([0, 1])

    activation_difference = handcrafted_activations - raw_activations
    loss = torch.sum(activation_difference * activation_difference)

    if grads:
        grads = torch.autograd.grad(loss, pattern, retain_graph=True)

    raw_extractor.release_hooks()
    handcrafted_extractor.release_hooks()

    return loss, grads


def trigger_attention_loss(raw_model, handed_model, backdoor_inputs, pattern, grads=True):
    raw_model.train()
    handed_activations = handed_model.first_activations(backdoor_inputs).mean([0, 1])
    raw_activations = raw_model.first_activations(backdoor_inputs).clone().detach().mean([0, 1])

    activation_difference = handed_activations - raw_activations
    loss = torch.sum(activation_difference * activation_difference)

    if grads:
        grads = torch.autograd.grad(loss, pattern, retain_graph=True)

    return loss, grads

def trigger_loss(model,backdoor_inputs, clean_inputs, pattern, grads=True):
    model.train()
    backdoor_activations = model.first_activations(backdoor_inputs).mean([0, 1])
    clean_activations = model.first_activations(clean_inputs).mean([0, 1])
    difference = backdoor_activations - clean_activations
    loss = torch.sum(difference * difference)
    
    if grads:
        grads = torch.autograd.grad(loss, pattern, retain_graph=True)

    return loss, grads


def compute_latent_cosine_similarity(params: Params,
                                     model: Model,
                                     fixed_model: Model,
                                     inputs,
                                     grads=None):
    if not fixed_model:
        return torch.tensor(0.0), None
    t = time.perf_counter()
    with torch.no_grad():
        _, fixed_latent = fixed_model(inputs, latent=True)
    _, latent = model(inputs)
    record_time(params, t, 'forward')

    loss = -torch.cosine_similarity(latent, fixed_latent).mean() + 1
    if grads:
        grads = get_grads(model, loss)

    return loss, grads


def compute_spectral_evasion_loss(params: Params,
                                  model: Model,
                                  fixed_model: Model,
                                  inputs,
                                  grads=None):
    """
    Evades spectral analysis defense. Aims to preserve the latent representation
    on non-backdoored inputs. Uses a checkpoint non-backdoored `fixed_model` to
    compare the outputs. Uses euclidean distance as penalty.


    :param params: training parameters
    :param model: current model
    :param fixed_model: saved non-backdoored model as a reference.
    :param inputs: training data inputs
    :param grads: compute gradients.

    :return:
    """

    if not fixed_model:
        return torch.tensor(0.0), None
    t = time.perf_counter()
    with torch.no_grad():
        _, fixed_latent = fixed_model(inputs, latent=True)
    _, latent = model(inputs, latent=True)
    record_time(params, t, 'latent_fixed')
    if params.spectral_similarity == 'norm':
        loss = torch.norm(latent - fixed_latent, dim=1).mean()
    elif params.spectral_similarity == 'cosine':
        loss = -torch.cosine_similarity(latent, fixed_latent).mean() + 1
    else:
        raise ValueError(f'Specify correct similarity metric for '
                         f'spectral evasion: [norm, cosine].')
    if grads:
        grads = get_grads(params, model, loss)

    return loss, grads


def get_latent_grads(backdoor_label, model, inputs, labels):
    model.eval()
    model.zero_grad()
    pred = model(inputs)
    z = torch.zeros_like(pred)

    z[list(range(labels.shape[0])), labels] = 1

    pred = pred * z
    pred.sum().backward(retain_graph=True)

    gradients = model.get_gradient()[labels == backdoor_label]
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3]).detach()
    model.zero_grad()

    return pooled_gradients


def sentinet_evasion_loss(backdoor_label, model, inputs, inputs_back, labels_back, grads):
    """The GradCam design is taken from:
    https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

    :param backdoor_label:
    :param params:
    :param model: 
    :param inputs: 
    :param inputs_back: 
    :param labels_back: 
    :param grads: 
    :return: 
    """
    pooled = get_latent_grads(backdoor_label, model, inputs, labels_back)
    t = time.perf_counter()
    features = model.features(inputs)
    features = features * pooled.view(1, 512, 1, 1)

    pooled_back = get_latent_grads(backdoor_label, model, inputs_back, labels_back)
    back_features = model.features(inputs_back)
    back_features = back_features * pooled_back.view(1, 512, 1, 1)

    features = torch.mean(features, dim=[0, 1], keepdim=True)
    features = F.relu(features) / features.max()

    back_features = torch.mean(back_features, dim=[0, 1], keepdim=True)
    back_features = F.relu(
        back_features) / back_features.max()
    loss = F.relu(back_features - features).max() * 10
    if grads:
        loss.backward(retain_graph=True)
        grads = copy_grad(model)

    return loss, grads


def norm_loss(nc_p_norm, model, grads=None):
    norm = None
    if nc_p_norm == 1:
        norm = torch.sum(th(model.mask))
    elif nc_p_norm == 2:
        norm = torch.norm(th(model.mask))
    else:
        raise ValueError('Not support mask norm.')

    if grads:
        grads = get_grads(model, norm)
        model.zero_grad()

    return norm, grads


def neural_cleanse_part1(nc_model, model, batch, backdoor_batch, nc_p_norm, nc_optim):
    nc_model.zero_grad()
    model.zero_grad()

    nc_model.switch_grads(True)
    model.switch_grads(False)

    output = model(nc_model(batch.inputs))
    nc_tasks = ['neural_cleanse_part1', 'mask_norm']

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    loss_values, grads = dict(), dict()
    for task in nc_tasks:
        if task == 'neural_cleanse_part1':
            loss_values[task], grads[task] = normal_loss(model, criterion, batch.inputs, backdoor_batch.labels,
                                                         grads=False)
        elif task == 'mask_norm':
            loss_values[task], grads[task] = norm_loss(nc_p_norm, model, grads=False)

    # logger.info(loss_values)
    # neural_cleanse_part1 and mark_norm not compute grads before
    loss = 0.999 * loss_values['neural_cleanse_part1'] + 0.001 * loss_values['mask_norm']
    loss.backward()
    nc_optim.step()

    nc_model.switch_grads(False)
    model.switch_grads(True)


def get_grads(model, loss):
    grads = list(torch.autograd.grad(loss.mean(),
                                     [x for x in model.parameters() if
                                      x.requires_grad],
                                     retain_graph=True))

    return grads


# UNTESTED
def estimate_fisher(params, model, data_loader, sample_size):
    # sample loglikelihoods from the dataset.
    loglikelihoods = []
    for x, y in data_loader:
        x = x.to(params.device)
        y = y.to(params.device)
        loglikelihoods.append(
            F.log_softmax(model(x)[0], dim=1)[range(params.batch_size), y]
        )
        if len(loglikelihoods) >= sample_size // params.batch_size:
            break
    # estimate the fisher information of the parameters.
    loglikelihoods = torch.cat(loglikelihoods).unbind()
    loglikelihood_grads = zip(*[autograd.grad(
        l, model.parameters(),
        retain_graph=(i < len(loglikelihoods))
    ) for i, l in enumerate(loglikelihoods, 1)])
    loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
    fisher_diagonals = [(g ** 2).mean(0) for g in loglikelihood_grads]
    param_names = [
        n.replace('.', '__') for n, p in model.named_parameters()
    ]
    return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}


def consolidate(model, fisher):
    for n, p in model.named_parameters():
        n = n.replace('.', '__')
        model.register_buffer('{}_mean'.format(n), p.data.clone())
        model.register_buffer('{}_fisher'
                              .format(n), fisher[n].data.clone())


def ewc_loss(params: Params, model: nn.Module, grads=None):
    try:
        losses = []
        for n, p in model.named_parameters():
            # retrieve the consolidated mean and fisher information.
            n = n.replace('.', '__')
            mean = getattr(model, '{}_mean'.format(n))
            fisher = getattr(model, '{}_fisher'.format(n))
            # wrap mean and fisher in variables.
            # calculate a ewc loss. (assumes the parameter's prior as
            # gaussian distribution with the estimated mean and the
            # estimated cramer-rao lower bound variance, which is
            # equivalent to the inverse of fisher information)
            losses.append((fisher * (p - mean) ** 2).sum())
        loss = (model.lamda / 2) * sum(losses)
        if grads:
            loss.backward()
            grads = get_grads(params, model, loss)
            return loss, grads
        else:
            return loss, None

    except AttributeError:
        # ewc loss is 0 if there's no consolidated parameters.
        print('exception')
        return torch.zeros(1).to(params.device), grads


def copy_grad(model: nn.Module):
    grads = list()
    for name, params in model.named_parameters():
        if not params.requires_grad:
            a = 1
            # print(name)
        else:
            grads.append(params.grad.clone().detach())
    model.zero_grad()
    return grads
