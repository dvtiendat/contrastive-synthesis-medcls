import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DINOLoss(nn.Module):
    """ Loss function for DINO self-supervised learning. """
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, total_epochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))

        warmup_steps = warmup_teacher_temp_epochs
        total_steps = total_epochs
        if warmup_steps >= total_steps: 
             self.teacher_temp_schedule = np.ones(total_steps) * teacher_temp
        else:
            self.teacher_temp_schedule = np.concatenate((
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_steps),
                np.ones(total_steps - warmup_steps) * teacher_temp
            ))


    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops) 

        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        # Detach teacher output so gradients don't flow back
        teacher_out = teacher_out.detach().chunk(2) 

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out): 
            for v in range(len(student_out)): 
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        if n_loss_terms > 0:
            total_loss /= n_loss_terms

        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update the center used for teacher output normalization (EMA).
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True) # Average over batch

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)