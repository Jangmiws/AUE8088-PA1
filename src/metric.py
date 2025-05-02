from torchmetrics import Metric
import torch

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert preds.shape == target.shape, "Predictions and targets must have the same shape"

        # [TODO] Cound the number of correct prediction
        correct = (preds == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()




class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('true_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        # preds: (batch_size, num_classes)
        preds = torch.argmax(preds, dim=1)

        for cls in range(self.num_classes):
            # True Positive
            tp = ((preds == cls) & (target == cls)).sum()

            # False Positive
            fp = ((preds == cls) & (target != cls)).sum()

            # False Negative
            fn = ((preds != cls) & (target == cls)).sum()

            self.true_positives[cls] += tp
            self.false_positives[cls] += fp
            self.false_negatives[cls] += fn

    def compute(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        
        return f1
    