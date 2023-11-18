import torch.nn as nn
import Models.DINO as module_vits
import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist

class DINO(nn.Module):
    """
    DINO model
    """
    def __init__(self, model_name, 
                patch_size=16,
                out_dim=65536,
                use_bn_in_head=False,
                norm_last_layer=True,
                drop_path_rate=0.1,
                image_size=224):

        if model_name in module_vits.__dict__.keys():
            student = module_vits.__dict__[model_name](
                patch_size=patch_size,
                drop_path_rate=drop_path_rate,  # stochastic depth
                img_size=image_size
            )
            teacher = module_vits.__dict__[model_name](patch_size=patch_size, 
                                                       img_size=image_size)
            self.embed_dim = student.embed_dim
        else:
            raise NotImplementedError(f"model {model_name} not supported")
        
        # multi-crop wrapper handles forward with inputs of different resolutions
        self.student = module_vits.MultiCropWrapper(
            student,
            module_vits.DINOHead(
                self.embed_dim,
                out_dim,
                use_bn=use_bn_in_head,
                norm_last_layer=norm_last_layer,
            ))
        self.teacher = module_vits.MultiCropWrapper(
            teacher,
            module_vits.DINOHead(self.embed_dim, out_dim, use_bn_in_head),
        )
    def get_model(self):
        return self.student, self.teacher, self.embed_dim


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum)

