import torch.nn.functional as F
import torch.nn as nn
class knowledge_distillation_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 1.0
        self.alpha = 0.5
    """
    计算知识蒸馏损失函数
    :param student_logits: 学生模型的输出 logits
    :param teacher_logits: 教师模型的输出 logits
    :param labels: 实际标签
    :param temperature: 温度参数，默认为 1.0
    :param alpha: 损失函数中交叉熵损失的权重，默认为 0.5
    :return: 知识蒸馏损失值
    """
    def forward(self,student_logits, teacher_logits, labels):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(student_logits, labels)

        # 计算 softmax 值
        student_probs = F.softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)

        # 计算 KL 散度损失
        kd_loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean') * self.temperature ** 2

        # 加权计算总损失
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
        return total_loss
