import numpy as np
import torch


# Code were written based on the sample code presented in the following paper:
# https://arxiv.org/pdf/1902.03680.pdf Learning From Noisy Labels By Regularized Estimation Of Annotator Confusion

def confusion_matrix_estimators(num_annotators, num_classes):
    """
    Defines confusion matrix estimators.
        This function defines a set of confusion matrices that characterize respective annotators.
        Here (i, j)th element in the annotator confusion matrix of annotator a is given by
        P(label_annotator_a = j| label_true = i) i.e. the probability that the annotator assigns label j to
        the image when the ground truth label is i.

    Args:
        num_annotators: Number of annotators
        num_classes: Number of classes.

    Returns:
        confusion_matrices: Annotator confusion matrices. A 'Tensor' of shape [num_annotators, num_classes, num_classes]
    """
    with torch.no_grad():
        # initialise so the confusion matrices are close to identity matrices.
        w_init = torch.tensor(
            np.stack([6.0 * np.eye(num_classes) - 5.0 for _ in range(num_annotators)]),
            dtype=torch.float32
        )
        rho = torch.nn.Parameter(w_init)

        # ensure positivity
        rho = torch.nn.functional.softplus(rho)

        # ensure each row sums to one
        confusion_matrices = torch.div(rho, torch.sum(rho, dim=-1, keepdim=True))

    return confusion_matrices


def cross_entropy_over_annotators(labels, logits, confusion_matrices):
    """ 
    Cross entropy between noisy labels from multiple annotators and their confusion matrix models.
    Args:
        labels: One-hot representation of labels from multiple annotators.
        torch.Tensor of size [batch, num_annotators, num_classes]. Missing labels are assumed to be
        represented as zero vectors.
        logits: Logits from the classifier. torch.Tensor of size [batch, num_classes]
        confusion_matrices: Confusion matrices of annotators. torch.Tensor of size
        [num_annotators, num_classes, num_classes]. The (i, j) th element of the confusion matrix
        for annotator a denotes the probability P(label_annotator_a = j|label_true = i).
    Returns:
        The average cross-entropy across annotators and image examples.
    """
    # Treat one-hot labels as probability vectors
    labels = labels.float()

    # Sequentially compute the loss for each annotator
    losses_all_annotators = []
    for idx, labels_annotator in enumerate(labels.unbind(dim=1)):
        loss = sparse_confusion_matrix_softmax_cross_entropy(
            labels=labels_annotator,
            logits=logits,
            confusion_matrix=confusion_matrices[idx, :, :]
        )
        losses_all_annotators.append(loss)

    # Stack them into a tensor of size (batch, num_annotators)
    losses_all_annotators = torch.stack(losses_all_annotators, dim=1)

    # Filter out annotator networks with no labels. This allows you train
    # annotator networks only when the labels are available.
    has_labels = labels.sum(dim=2) # (batch, num_annotators)
    losses_all_annotators = losses_all_annotators * has_labels

    return torch.mean(losses_all_annotators.sum(dim=1))


def sparse_confusion_matrix_softmax_cross_entropy(labels, logits, confusion_matrix):
    """
    Cross entropy between noisy labels and confusion matrix based model for a single annotator.
    Args:
        labels: One-hot representation of labels. Tensor of size [batch, num_classes].
        logits: Logits from the classifier. Tensor of size [batch, num_classes]
        confusion_matrix: Confusion matrix of the annotator. Tensor of size [num_classes, num_classes].
    Returns:
        The average cross-entropy across annotators for image examples
        Returns a 'Tensor' of size [batch_size].
    """
    # get the predicted label distribution
    preds_true = torch.nn.functional.softmax(logits, dim=1)

    # Map label distribution into annotator label distribution by
    # multiplying it by its confusion matrix.
    preds_annotator = torch.matmul(preds_true, confusion_matrix)

    # cross entropy
    preds_clipped = torch.clamp(preds_annotator, 1e-10, 0.9999999)
    cross_entropy = torch.sum(-labels * torch.log(preds_clipped), dim=1)

    return cross_entropy