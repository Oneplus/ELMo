import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


class SoftmaxLayer(torch.nn.Module):
  """ Naive softmax-layer """
  def __init__(self, input_dim: int, n_class: int):
    """

    :param input_dim: int
    :param n_class: int
    """
    super(SoftmaxLayer, self).__init__()
    self.hidden2tag = torch.nn.Linear(input_dim, n_class)
    self.criterion = torch.nn.CrossEntropyLoss(size_average=False)

  def forward(self, embeddings: torch.Tensor, targets: torch.Tensor):
    """

    :param embeddings: torch.Tensor
    :param targets: torch.Tensor
    :return:
    """
    tag_scores = self.hidden2tag(embeddings)
    return self.criterion(tag_scores, targets)
