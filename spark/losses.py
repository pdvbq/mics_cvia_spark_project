"""
Loss functions implementations for SPARK project

"""
import torch

def generalized_iou_loss(pred_boxes, target_boxes):
    """
    Compute the Generalized IoU (GIoU) loss between predicted and target bounding boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes, shape (N, 4).
                            Format: [x_min, y_min, x_max, y_max]
        target_boxes (Tensor): Ground truth bounding boxes, shape (N, 4).
                               Format: [x_min, y_min, x_max, y_max]

    Returns:
        Tensor: GIoU loss for each pair of bounding boxes, shape (N,).
    """
    # Calculate intersection
    inter_xmin = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_ymin = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_xmax = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_ymax = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = (inter_xmax - inter_xmin).clamp(min=0) * (inter_ymax - inter_ymin).clamp(min=0)

    # Calculate areas of individual boxes
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    # Union area
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / union_area.clamp(min=1e-6)

    # Find the smallest enclosing box
    enclosing_xmin = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclosing_ymin = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclosing_xmax = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclosing_ymax = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclosing_area = (enclosing_xmax - enclosing_xmin).clamp(min=0) * (enclosing_ymax - enclosing_ymin).clamp(min=0)

    # Generalized IoU
    giou = iou - (enclosing_area - union_area) / enclosing_area.clamp(min=1e-6)

    # GIoU loss
    giou_loss = 1 - giou

    return giou_loss

def distance_iou_loss(pred_boxes, target_boxes):
    """
    Compute the Distance IoU (DIoU) loss between predicted and target bounding boxes.

    Args:
        pred_boxes (Tensor): Predicted bounding boxes, shape (N, 4).
                            Format: [x_min, y_min, x_max, y_max]
        target_boxes (Tensor): Ground truth bounding boxes, shape (N, 4).
                               Format: [x_min, y_min, x_max, y_max]

    Returns:
        Tensor: DIoU loss for each pair of bounding boxes, shape (N,).
    """
    # Calculate intersection
    inter_xmin = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_ymin = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_xmax = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_ymax = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = (inter_xmax - inter_xmin).clamp(min=0) * (inter_ymax - inter_ymin).clamp(min=0)

    # Calculate areas of individual boxes
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    # Union area
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / union_area.clamp(min=1e-6)

    # Center coordinates of predicted and target boxes
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    # Squared Euclidean distance between box centers
    center_distance = (pred_center_x - target_center_x).pow(2) + (pred_center_y - target_center_y).pow(2)

    # Find the smallest enclosing box
    enclosing_xmin = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclosing_ymin = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclosing_xmax = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclosing_ymax = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclosing_diagonal = (enclosing_xmax - enclosing_xmin).pow(2) + (enclosing_ymax - enclosing_ymin).pow(2)

    # Distance IoU
    diou = iou - center_distance / enclosing_diagonal.clamp(min=1e-6)

    # DIoU loss
    diou_loss = 1 - diou

    return diou_loss

def balanced_l1_loss(pred, target, beta=1.0):
    """
    Compute the Balanced L1 Loss as described in the Libra R-CNN paper.

    Args:
        pred (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        beta (float): Transition point between L1 and L2 loss. Default is 1.0.

    Returns:
        Tensor: Balanced L1 loss.
    """
    diff = torch.abs(pred - target)
    b = torch.exp(beta) - 1
    loss = torch.where(
        diff < 1.0 / beta,
        beta * diff,
        torch.log(diff * beta + 1) / b
    )
    return loss

def smooth_l1_loss(pred, target, beta=1.0):
    """
    Compute the Smooth L1 Loss, commonly used in object detection.

    Args:
        pred (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        beta (float): Transition point between L1 and L2 loss. Default is 1.0.

    Returns:
        Tensor: Smooth L1 loss.
    """
    diff = torch.abs(pred - target)
    loss = torch.where(
        diff < beta,
        0.5 * (diff ** 2) / beta,
        diff - 0.5 * beta
    )
    return loss