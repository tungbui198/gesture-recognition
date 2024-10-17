from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import InstanceData

from src.registry import TRANSFORMS
from src.utils import ActionDataSample


@TRANSFORMS.register_module()
class PackActionInputs(BaseTransform):
    """Pack the inputs data.

    Args:
        collect_keys (tuple[str], optional): The keys to be collected
            to ``packed_results['inputs']``. Defaults to ``
        meta_keys (Sequence[str]): The meta keys to saved in the
            `metainfo` of the `data_sample`.
            Defaults to ``('img_shape', 'img_key', 'video_id', 'timestamp')``.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
    """

    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_labels": "labels",
    }

    def __init__(
        self,
        collect_keys: Optional[Tuple[str]] = None,
        meta_keys: Sequence[str] = ("img_shape", "img_key", "video_id", "timestamp"),
        algorithm_keys: Sequence[str] = (),
    ) -> None:
        self.collect_keys = collect_keys
        self.meta_keys = meta_keys
        self.algorithm_keys = algorithm_keys

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PackActionInputs`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        packed_results = dict()
        if self.collect_keys is not None:
            packed_results["inputs"] = dict()
            for key in self.collect_keys:
                packed_results["inputs"][key] = to_tensor(results[key])
        else:
            if "imgs" in results:
                imgs = results["imgs"]
                packed_results["inputs"] = to_tensor(imgs)
            elif "heatmap_imgs" in results:
                heatmap_imgs = results["heatmap_imgs"]
                packed_results["inputs"] = to_tensor(heatmap_imgs)
            elif "keypoint" in results:
                keypoint = results["keypoint"]
                packed_results["inputs"] = to_tensor(keypoint)
            elif "audios" in results:
                audios = results["audios"]
                packed_results["inputs"] = to_tensor(audios)
            elif "text" in results:
                text = results["text"]
                packed_results["inputs"] = to_tensor(text)
            else:
                raise ValueError(
                    "Cannot get `imgs`, `keypoint`, `heatmap_imgs`, "
                    "`audios` or `text` in the input dict of "
                    "`PackActionInputs`."
                )

        data_sample = ActionDataSample()

        if "gt_bboxes" in results:
            instance_data = InstanceData()
            for key in self.mapping_table.keys():
                instance_data[self.mapping_table[key]] = to_tensor(results[key])
            data_sample.gt_instances = instance_data

            if "proposals" in results:
                data_sample.proposals = InstanceData(
                    bboxes=to_tensor(results["proposals"])
                )

        if "label" in results:
            data_sample.set_gt_label(results["label"])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(collect_keys={self.collect_keys}, "
        repr_str += f"meta_keys={self.meta_keys})"
        return repr_str


@TRANSFORMS.register_module()
class FormatGCNInput(BaseTransform):
    """Format final skeleton shape.

    Required Keys:

        - keypoint
        - keypoint_score (optional)
        - num_clips (optional)

    Modified Key:

        - keypoint

    Args:
        num_person (int): The maximum number of people. Defaults to 2.
        mode (str): The padding mode. Defaults to ``'zero'``.
    """

    def __init__(self, num_person: int = 2, mode: str = "zero") -> None:
        self.num_person = num_person
        assert mode in ["zero", "loop"]
        self.mode = mode

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`FormatGCNInput`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        keypoint = results["keypoint"]
        if "keypoint_score" in results:
            keypoint = np.concatenate(
                (keypoint, results["keypoint_score"][..., None]), axis=-1
            )

        cur_num_person = keypoint.shape[0]
        if cur_num_person < self.num_person:
            pad_dim = self.num_person - cur_num_person
            pad = np.zeros((pad_dim,) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == "loop" and cur_num_person == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif cur_num_person > self.num_person:
            keypoint = keypoint[: self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get("num_clips", 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)

        results["keypoint"] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self) -> str:
        repr_str = (
            f"{self.__class__.__name__}("
            f"num_person={self.num_person}, "
            f"mode={self.mode})"
        )
        return repr_str
