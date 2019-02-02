import torch


class SoftmaxLoss(torch.nn.Module):

    def __init__(self, input_dim: int, n_class: int, pad_id: int = 1):
        super(SoftmaxLoss, self).__init__()
        self.projection = torch.nn.Linear(input_dim, n_class)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, size_average=False)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.projection)
        torch.nn.init.constant_(self.projection.bias, 0.)

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor):
        tag_scores = self.projection(embeddings)
        return self.criterion(tag_scores, targets)
