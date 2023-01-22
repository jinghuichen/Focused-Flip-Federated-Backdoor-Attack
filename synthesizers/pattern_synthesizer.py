import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from FederatedTask import Cifar10FederatedTask,TinyImagenetFederatedTask
from FederatedTask import FederatedTask

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class PatternSynthesizer(Synthesizer):
    pattern_tensor: torch.Tensor = torch.tensor([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ])
    "Just some random 2D pattern."

    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 23
    "Y coordinate to put the backdoor into."

    x_bot = x_top + pattern_tensor.shape[0]
    y_bot = y_top + pattern_tensor.shape[1]

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."
    dbas = []

    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, task: FederatedTask, handcraft, distributed, mal: tuple):
        super().__init__(task)
        self.i_mal = mal[0]
        self.n_mal = mal[1]
        self.make_pattern(task, self.pattern_tensor, self.x_top, self.y_top, handcraft)
        if distributed and self.i_mal != 0:
            self.random_break_trigger(task)

    def make_pattern(self, task,pattern_tensor, x_top, y_top, handcraft):
        if isinstance(task,Cifar10FederatedTask):
            trigger_size = (3,3)
        elif isinstance(task, TinyImagenetFederatedTask):
            trigger_size = (4,4)
        if handcraft:
            torch.manual_seed(111)
            pattern_tensor = torch.rand(trigger_size)
            print('Initial Tensor:\n', pattern_tensor)
            pattern_tensor = (pattern_tensor * 255).floor() / 255
            self.x_bot = x_top + pattern_tensor.shape[0]
            self.y_bot = y_top + pattern_tensor.shape[1]
        else:
            pattern_tensor = torch.zeros(trigger_size)
            print('Initial Tensor:\n', pattern_tensor)
            self.x_bot = x_top + pattern_tensor.shape[0]
            self.y_bot = y_top + pattern_tensor.shape[1]

        full_image = torch.zeros(self.params.input_shape).fill_(self.mask_value)

        x_bot = self.x_bot
        y_bot = self.y_bot

        # mask is 1 when the pattern is presented
        if x_bot >= self.params.input_shape[1] or y_bot >= self.params.input_shape[2]:
            raise ValueError(f'Position of backdoor outside image limits:'
                             f'image: {self.params.input_shape}, but backdoor'
                             f'ends at ({x_bot}, {y_bot})')

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)
        self.pattern = self.task.normalize(full_image).to(self.params.device)

    def random_break_trigger(self,task):
        x_top, y_top = self.x_top, self.y_top
        i_mal, n_mal = self.i_mal, self.n_mal
        assert (n_mal in [1, 2, 4])
        if n_mal == 1:
            if isinstance(task,Cifar10FederatedTask):
                for p in range(3):
                    gx = random.randint(0, 2)
                    gy = random.randint(0, 2)
                    self.mask[:,x_top + gx, y_top + gy] = 0
            elif isinstance(task, TinyImagenetFederatedTask):
                for p in range(9):
                    gx = random.randint(0, 3)
                    gy = random.randint(0, 3)
                    self.mask[:,x_top + gx, y_top + gy] = 0
        elif n_mal == 2:
            if i_mal == 1:
                if isinstance(task,Cifar10FederatedTask):
                    self.mask[:,x_top, y_top] = 0
                    self.mask[:,x_top + 2, y_top] = 0
                    self.mask[:,x_top + 2, y_top] = 0
                    self.mask[:,x_top + 2, y_top + 2] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:,x_top, y_top] = 0
                    self.mask[:,x_top+3, y_top] = 0
                    self.mask[:,x_top, y_top+3] = 0
                    self.mask[:,x_top+3, y_top+3] = 0
            elif i_mal == 2:
                if isinstance(task,Cifar10FederatedTask):
                    self.mask[:,x_top, y_top + 1] = 0
                    self.mask[:,x_top + 2, y_top + 1] = 0
                    self.mask[:,x_top + 1, y_top] = 0
                    self.mask[:,x_top + 1, y_top + 2] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:,x_top, y_top+1:y_top+3] = 0
                    self.mask[:,x_top+3, y_top+1:y_top+3] = 0
                    self.mask[:,x_top+1:x_top+3, y_top] = 0
                    self.mask[:,x_top+1:x_top+3, y_top+3] = 0
            else:
                raise ValueError("out of mal index!")
            print("dba mask:{}:\n".format((i_mal,n_mal)),self.mask[0,3:7,23:27])
        elif n_mal == 4:
            if i_mal == 1:
                if isinstance(task,Cifar10FederatedTask):
                    self.mask[:,x_top, y_top] = 0
                    self.mask[:,x_top + 1, y_top] = 0
                    self.mask[:,x_top, y_top + 1] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:,x_top+3,y_top:y_top+4]=0
                    self.mask[:,x_top:x_top+4,y_top+3]=0
            if i_mal == 2:
                if isinstance(task,Cifar10FederatedTask):
                    self.mask[:,x_top, y_top + 2] = 0
                    self.mask[:,x_top + 1, y_top + 2] = 0
                    self.mask[:,x_top, y_top + 1] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:,x_top:x_top+4,y_top]=0
                    self.mask[:,x_top+3,y_top:y_top+4]=0
            if i_mal == 3:
                if isinstance(task,Cifar10FederatedTask):
                    self.mask[:,x_top + 2, y_top] = 0
                    self.mask[:,x_top + 2, y_top + 1] = 0
                    self.mask[:,x_top + 1, y_top + 0] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:,x_top,y_top:y_top+4]=0
                    self.mask[:,x_top:x_top+4,y_top+3]=0
            if i_mal == 4:
                if isinstance(task,Cifar10FederatedTask):
                    self.mask[:,x_top + 2, y_top + 2] = 0
                    self.mask[:,x_top + 1, y_top + 2] = 0
                    self.mask[:,x_top + 2, y_top + 1] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:,x_top:x_top+4,y_top]=0
                    self.mask[:,x_top,y_top:y_top+4]=0
            print("dba mask:{}:\n".format((i_mal,n_mal)),self.mask[0,x_top:x_top+4,y_top:y_top+4])
        else:
            raise ValueError("Not implement DBA for num of clients out of 1,2,4")

    def synthesize_inputs(self, batch, attack_portion=None):
        pattern, mask = self.get_pattern()
        batch.inputs[:attack_portion] = (1 - mask) * batch.inputs[:attack_portion] + mask * pattern

        return

    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)

        return

    def get_pattern(self):
        if self.params.backdoor_dynamic_position:
            resize = random.randint(self.resize_scale[0], self.resize_scale[1])
            pattern = self.pattern_tensor
            if random.random() > 0.5:
                pattern = functional.hflip(pattern)
            image = transform_to_image(pattern)
            pattern = transform_to_tensor(
                functional.resize(image,
                                  resize, interpolation=0)).squeeze()

            x = random.randint(0, self.params.input_shape[1] - pattern.shape[0] - 1)
            y = random.randint(0, self.params.input_shape[2] - pattern.shape[1] - 1)
            self.make_pattern(pattern, x, y)

        return self.pattern, self.mask