import torch
from metrics.metric import Metric


class TestLossMetric(Metric):

    def __init__(self, criterion, train=False):
        self.criterion  = criterion
        self.main_metric_name = 'loss_value'
        super().__init__(name='Loss', train=False)

    def compute_metric(self, outputs: torch.Tensor,
                       labels: torch.Tensor, top_k=(1,)):
        """Computes the precision@k for the specified values of k"""
        loss = None
        # print("torch.sum:{}".format(torch.sum(outputs[0])))
        if 0.999<=torch.sum(outputs[0])<=1.001:
            # print("Input is softmax")
            outputs=torch.log(outputs)
            nllloss_func=torch.nn.NLLLoss()
            loss = nllloss_func(outputs,labels)
        else:
            loss = self.criterion(outputs, labels)
        return {'loss_value': loss.mean().item()}