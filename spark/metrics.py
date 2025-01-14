"""
Metrics implementations for SPARK project

"""
import torch
from torch import Tensor


class Precision:
    def __init__(self, average: str = "macro"):
        """
        Initializes the Precision metric.

        Args:
            average (str): Method to calculate precision for multi-class data.
                - 'macro': Calculate precision for each class, then take the average.
                - 'weighted': Calculate precision for each class, weighted by class frequency.
                - 'none': Return precision for each class separately.
        """
        self.average = average
    

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        r""" This metric measures the precision based on the True Positive values
        and the False Positive values.

        The precision is computed as:

        ..math::
            \frac{\text{True Positives (TP)}}{\text{True Positives (TP) + False Positives (FP)}}

        Args:
            y_true (Tensor): Ground truth integer labels (0s, 1s, ...).
            y_pred (Tensor): Predicted integer labels (0s, 1s, ...).

        Examples::

            >>> y_true = torch.tensor([1,0,1,1,0,1,0]) # Ground truth values
            >>> y_pred = torch.tensor([1,0,1,0,0,1,1]) # Predicted values
            >>> 
            >>> precision_metric = Precision("macro")
            >>> 
            >>> precision = precision_metric(y_true, y_pred)
        """
        # Ensure inputs are integer
        y_true = y_true.int()
        y_pred = y_pred.int()

        # Get the unique classes
        classes = torch.unique(torch.cat([y_true, y_pred]))
        class_precisions = []

        for cls in classes:
            # Calculate true positives and false positives for the current class
            true_positive = torch.sum((y_pred == cls) & (y_true == cls)).float()
            false_positive = torch.sum((y_pred == cls) & (y_true != cls)).float()

            # Avoid division by zero
            if true_positive + false_positive == 0:
                class_precision = torch.tensor(0.0)
            else:
                class_precision = true_positive / (true_positive + false_positive)

            class_precisions.append(class_precision)

        class_precisions = torch.tensor(class_precisions)

        # Handle different averaging methods
        if self.average == "macro":
            return class_precisions.mean()
        elif self.average == "weighted":
            class_counts = torch.tensor([(y_true == cls).sum().item() for cls in classes]).float()
            return torch.sum(class_precisions * class_counts / class_counts.sum())
        elif self.average == "none":
            return class_precisions
        else:
            raise ValueError("Unsupported average type. Use 'macro', 'weighted', or 'none'.")


class Recall:
    def __init__(self, average: str = "macro"):
        """
        Initializes the Recall metric.

        Args:
            average (str): Method to calculate recall for multi-class data.
                - 'macro': Calculate recall for each class, then take the average.
                - 'weighted': Calculate recall for each class, weighted by class frequency.
                - 'none': Return recall for each class separately.
        """
        self.average = average
    

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        r""" This metric measures the recall based on the True Positive values
        and the False Negative values.

        The recall is computed as:

        ..math::
            \frac{\text{True Positives (TP)}}{\text{True Positives (TP) + False Negatives (FN)}}

        Args:
            y_true (Tensor): Ground truth integer labels (0s, 1s, ...).
            y_pred (Tensor): Predicted integer labels (0s, 1s, ...).

        Examples::

            >>> y_true = torch.tensor([1,0,1,1,0,1,0]) # Ground truth values
            >>> y_pred = torch.tensor([1,0,1,0,0,1,1]) # Predicted values
            >>> 
            >>> recall_metric = Recall("macro")
            >>> 
            >>> recall = recall_metric(y_true, y_pred)
        """
        # Ensure inputs are integer
        y_true = y_true.int()
        y_pred = y_pred.int()

        # Get the unique classes
        classes = torch.unique(torch.cat([y_true, y_pred]))
        class_recalls = []

        for cls in classes:
            # Calculate true positives and false negatives for the current class
            true_positive = torch.sum((y_pred == cls) & (y_true == cls)).float()
            false_negative = torch.sum((y_pred != cls) & (y_true == cls)).float()

            # Avoid division by zero
            if true_positive + false_negative == 0:
                class_recall = torch.tensor(0.0)
            else:
                class_recall = true_positive / (true_positive + false_negative)

            class_recalls.append(class_recall)

        class_recalls = torch.tensor(class_recalls)

        # Handle different averaging methods
        if self.average == "macro":
            return class_recalls.mean()
        elif self.average == "weighted":
            class_counts = torch.tensor([(y_true == cls).sum().item() for cls in classes]).float()
            return torch.sum(class_recalls * class_counts / class_counts.sum())
        elif self.average == "none":
            return class_recalls
        else:
            raise ValueError("Unsupported average type. Use 'macro', 'weighted', or 'none'.")


class FScore:
    def __init__(self, average: str = "macro", beta: float = 2.0):
        """
        Initializes the FScore metric.

        Args:
            average (str): Method to calculate FScore for multi-class data.
                - 'macro': Calculate FScore for each class, then take the average.
                - 'weighted': Calculate FScore for each class, weighted by class frequency.
                - 'none': Return FScore for each class separately.
            beta (float): Weight of recall in the FScore calculation. Default is 2 (F2 score).
        """
        self.average = average
        self.beta = beta
    

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        r""" This metric measures the f\beta score based on the Precision and 
        the Recall.

        The f\beta score is computed as:

        ..math::
            (1 + \beta²) \cdot \frac{\text{True Positives (TP)}}{\beta² \cdot \text{True Positives
             (TP) + False Negatives (FN)}}

        Args:
            y_true (Tensor): Ground truth integer labels (0s, 1s, ...).
            y_pred (Tensor): Predicted integer labels (0s, 1s, ...).

        Examples::

            >>> # Example of computing f2 score
            >>> y_true = torch.tensor([1,0,1,1,0,1,0]) # Ground truth values
            >>> y_pred = torch.tensor([1,0,1,0,0,1,1]) # Predicted values
            >>> beta = 1 # beta value
            >>> 
            >>> f2_score_metric = FScore()
            >>> 
            >>> f2_score = f2_score_metric(y_true, y_pred)

            >>> # Example of computing f1 score
            >>> y_true = torch.tensor([1,0,1,1,0,1,0]) # Ground truth values
            >>> y_pred = torch.tensor([1,0,1,0,0,1,1]) # Predicted values
            >>> 
            >>> f1_score_metric = FScore(beta)
            >>> 
            >>> f1_score = f1_score_metric(y_true, y_pred)

        """
        # Ensure inputs are integer
        y_true = y_true.int()
        y_pred = y_pred.int()

        # Get the unique classes
        classes = torch.unique(torch.cat([y_true, y_pred]))
        class_fscores = []

        for cls in classes:
            # Calculate true positives, false positives, and false negatives for the current class
            true_positive = torch.sum((y_pred == cls) & (y_true == cls)).float()
            false_positive = torch.sum((y_pred == cls) & (y_true != cls)).float()
            false_negative = torch.sum((y_pred != cls) & (y_true == cls)).float()

            # Avoid division by zero
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

            if precision + recall == 0:
                fscore = torch.tensor(0.0)
            else:
                fscore = (1 + self.beta ** 2) * (precision * recall) / ((self.beta ** 2) * precision + recall)

            class_fscores.append(fscore)

        class_fscores = torch.tensor(class_fscores)

        # Handle different averaging methods
        if self.average == "macro":
            return class_fscores.mean()
        elif self.average == "weighted":
            class_counts = torch.tensor([(y_true == cls).sum().item() for cls in classes]).float()
            return torch.sum(class_fscores * class_counts / class_counts.sum())
        elif self.average == "none":
            return class_fscores
        else:
            raise ValueError("Unsupported average type. Use 'macro', 'weighted', or 'none'.")


class IoU:
    def __init__(self, form: str = "default"):
        """
        Initializes the IoU metric class with a specific bounding box format.

        Args:
            form (str): Format of the coordinates:
                - 'default': [x_min, y_min, x_max, y_max]
                - 'YOLO': [x_center, y_center, width, height]
        """
        if form not in {"default", "YOLO"}:
            raise ValueError("Invalid form. Must be 'default' or 'YOLO'.")
        self.form = form
    

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        r""" This metric measures the intersection over union based on the area of 
        intersection and area of union of the predicted image and the ground truth image.

        The intersection over union is computed as:

        ..math::
            \frac{\text{Area of Intersection}}{\text{Area of Union}}

        Args:
            y_true (Tensor): Ground truth coordinates.
            y_pred (Tensor): Predicted coordinates.

        Examples::

            >>> y_true = torch.tensor([[2,3,4,2],[2,2.5,4,5]]) # Ground truth values
            >>> y_pred = torch.tensor([[4,2.5,4,5],[2,3,4,2]]) # Predicted values
            >>> form = "YOLO" # yolo format
            >>> 
            >>> iou_metric = IoU(form)
            >>> 
            >>> iou = iou_metric(y_true, y_pred)
        """

        # Ensure input is float32
        y_true = y_true.to(torch.float32)
        y_pred = y_pred.to(torch.float32)

        # Convert from YOLO to default format if necessary
        if self.form == "YOLO":
            y_true = self._convert_yolo_to_default(y_true)
            y_pred = self._convert_yolo_to_default(y_pred)

        # Compute the intersection coordinates
        intersection_x_min = torch.max(y_true[:, 0], y_pred[:, 0])
        intersection_y_min = torch.max(y_true[:, 1], y_pred[:, 1])
        intersection_x_max = torch.min(y_true[:, 2], y_pred[:, 2])
        intersection_y_max = torch.min(y_true[:, 3], y_pred[:, 3])

        # Compute the intersection area
        intersection_width = torch.clamp(intersection_x_max - intersection_x_min, min=0)
        intersection_height = torch.clamp(intersection_y_max - intersection_y_min, min=0)
        intersection_area = intersection_width * intersection_height

        # Compute the area of each box
        area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
        area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])

        # Compute the union area
        union_area = area_true + area_pred - intersection_area

        # Compute IoU
        iou = intersection_area / union_area
        iou = torch.where(union_area == 0, torch.tensor(0.0, dtype=torch.float32), iou)

        return iou

    def _convert_yolo_to_default(self, boxes: Tensor) -> Tensor:
        """
        Convert YOLO format [x_center, y_center, width, height] to 
        default format [x_min, y_min, x_max, y_max].

        Args:
            boxes (Tensor): Bounding boxes in YOLO format.

        Returns:
            Tensor: Bounding boxes in default format.
        """
        x_min = boxes[:, 0] - boxes[:, 2] / 2
        y_min = boxes[:, 1] - boxes[:, 3] / 2
        x_max = boxes[:, 0] + boxes[:, 2] / 2
        y_max = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)


class MAPK:
    def __init__(self, k: int = 2):
        """
        Initialize the MAPK metric.

        Args:
            k (int): Threshold value to consider top-k predictions. Default is 2.
        """
        self.k = k

    def __call__(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        r""" This metric measures the mean average precision at a threshold K (mAP).

        The mAP is computed as:

        ..math::
            Precision(k) = \frac{\text{Number of relevant items in top k results}}{k}
            \frac{1}{n relevant items} \sum_{k=1}^N{Precision(k) \cdot relevant(k)}

        Args:
            y_true (Tensor): Ground truth binary labels (0s and 1s).
            y_pred (Tensor): Predicted relevance scores or binary labels.

        Examples::

            >>> y_true = torch.tensor([1,0,1,1,0,1,0]) # Ground truth values
            >>> y_pred = torch.tensor([0.9,0.4,0.95,0.3,0.2,0.8,0.1]) # Predicted scores
            >>> k = 3 # threshold value
            >>> 
            >>> mapk_metric = MAPK(k)
            >>> 
            >>> mapk = mapk_metric(y_true.unsqueeze(0), y_pred.unsqueeze(0))

        """
        # Ensure ground truth values are binary
        y_true = y_true.int()

        ap_list = []

        # Compute AP@k for each query
        for truth, pred in zip(y_true, y_pred):
            ap_at_k = self.compute_ap_at_k(truth, pred, self.k)
            ap_list.append(ap_at_k)

        # Compute the mean of AP values
        map_at_k = torch.tensor(ap_list).mean()
        return map_at_k

    def compute_precision_at_k(self, y_true: Tensor, y_pred: Tensor, k: int) -> float:
        """
        Computes Precision@k given ground truth and predicted values.

        Args:
            y_true (Tensor): Ground truth binary labels (0s and 1s).
            y_pred (Tensor): Predicted relevance scores or binary labels.
            k (int): Threshold value to consider top-k predictions.
        """
        y_true = y_true[:k]
        y_pred = y_pred[:k]

        # Compute number of relevant and retrieved items
        num_relevant = torch.sum(y_true)
        num_retrieved = len(y_pred)

        # Avoid division by zero
        if num_retrieved == 0:
            return 0.0

        # Compute precision
        precision_at_k = num_relevant / num_retrieved
        return precision_at_k.item()

    def compute_ap_at_k(self, y_true: Tensor, y_pred: Tensor, k: int) -> float:
        """
        Computes the Average Precision@k for a single query.

        Args:
            y_true (Tensor): Ground truth binary labels (0s and 1s).
            y_pred (Tensor): Predicted relevance scores or binary labels.
            k (int): Threshold value to consider top-k predictions.
        """
        # Sort predictions and corresponding ground truth
        _, indices = torch.sort(y_pred, descending=True)
        y_true = y_true[indices]

        ap_at_k = 0.0
        num_relevant = torch.sum(y_true).item()

        # Avoid division by zero if there are no relevant items
        if num_relevant == 0:
            return 0.0

        # Compute AP@k
        for i in range(min(k, len(y_pred))):
            if y_true[i] == 1:
                precision_at_i = self.compute_precision_at_k(y_true, y_pred, i + 1)
                ap_at_k += precision_at_i

        ap_at_k /= num_relevant
        return ap_at_k