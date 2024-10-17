import copy
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from src.registry import METRICS


def confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ["true", "pred", "all", None]:
        raise ValueError("normalize must be one of {'true', 'pred', " "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
        if y_pred.dtype == np.int32:
            y_pred = y_pred.astype(np.int64)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f"y_pred must be list or np.ndarray, but got {type(y_pred)}")
    if not y_pred.dtype == np.int64:
        raise TypeError(f"y_pred dtype must be np.int64, but got {y_pred.dtype}")

    if isinstance(y_real, list):
        y_real = np.array(y_real)
        if y_real.dtype == np.int32:
            y_real = y_real.astype(np.int64)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(f"y_real must be list or np.ndarray, but got {type(y_real)}")
    if not y_real.dtype == np.int64:
        raise TypeError(f"y_real dtype must be np.int64, but got {y_real.dtype}")

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped, minlength=num_labels**2
    ).reshape(num_labels, num_labels)

    with np.errstate(all="ignore"):
        if normalize == "true":
            confusion_mat = confusion_mat / confusion_mat.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            confusion_mat = confusion_mat / confusion_mat.sum(axis=0, keepdims=True)
        elif normalize == "all":
            confusion_mat = confusion_mat / confusion_mat.sum()
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat


def top_k_accuracy(scores, labels, topk=(1,)):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


def mean_class_accuracy(scores, labels):
    """Calculate mean class accuracy.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.

    Returns:
        np.ndarray: Mean class accuracy.
    """
    pred = np.argmax(scores, axis=1)
    cf_mat = confusion_matrix(pred, labels).astype(float)

    cls_cnt = cf_mat.sum(axis=1)
    cls_hit = np.diag(cf_mat)

    mean_class_acc = np.mean(
        [hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)]
    )

    return mean_class_acc


@METRICS.register_module()
class AccMetric(BaseMetric):
    """Accuracy evaluation metric."""

    default_prefix: Optional[str] = "acc"

    def __init__(
        self,
        metric_list: Optional[Union[str, Tuple[str]]] = (
            "top_k_accuracy",
            "mean_class_accuracy",
        ),
        collect_device: str = "cpu",
        metric_options: Optional[Dict] = dict(top_k_accuracy=dict(topk=(1, 5))),
        prefix: Optional[str] = None,
    ) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError(
                "metric_list must be str or tuple of str, "
                f"but got {type(metric_list)}"
            )

        if isinstance(metric_list, str):
            metrics = (metric_list,)
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in ["top_k_accuracy", "mean_class_accuracy"]

        self.metrics = metrics
        self.metric_options = metric_options

    def process(
        self, data_batch: Sequence[Tuple[Any, Dict]], data_samples: Sequence[Dict]
    ) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample["pred_score"]
            label = data_sample["gt_label"]

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result["pred"] = pred
            if label.size(0) == 1:
                # single-label
                result["label"] = label.item()
            else:
                # multi-label
                result["label"] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x["label"] for x in results]
        preds = [x["pred"] for x in results]
        return self.calculate(preds, labels)

    def calculate(
        self, preds: List[np.ndarray], labels: List[Union[int, np.ndarray]]
    ) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        metric_options = copy.deepcopy(self.metric_options)
        for metric in self.metrics:
            if metric == "top_k_accuracy":
                topk = metric_options.setdefault("top_k_accuracy", {}).setdefault(
                    "topk", (1, 5)
                )

                if not isinstance(topk, (int, tuple)):
                    raise TypeError(
                        "topk must be int or tuple of int, " f"but got {type(topk)}"
                    )

                if isinstance(topk, int):
                    topk = (topk,)

                top_k_acc = top_k_accuracy(preds, labels, topk)
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f"top{k}"] = acc

            if metric == "mean_class_accuracy":
                mean1 = mean_class_accuracy(preds, labels)
                eval_results["mean1"] = mean1

        return eval_results
