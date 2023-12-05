# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import numpy as np

from otx.api.entities.result_media import ResultMediaEntity


torch.manual_seed(0)

assert_text_explain_all = "The number of saliency maps should be equal to the number of all classes."
assert_text_explain_predicted = "The number of saliency maps should be equal to the number of predicted classes."


def saliency_maps_check(
    predicted_dataset, task_labels, raw_sal_map_shape=None, processed_saliency_maps=False, only_predicted=True
):
    for data_point in predicted_dataset:
        saliency_map_counter = 0
        metadata_list = data_point.get_metadata()
        for metadata in metadata_list:
            if isinstance(metadata.data, ResultMediaEntity):
                if metadata.data.type == "saliency_map":
                    saliency_map_counter += 1
                    if processed_saliency_maps:
                        assert metadata.data.numpy.ndim == 3, "Number of dims is incorrect."
                        assert metadata.data.numpy.shape == (data_point.height, data_point.width, 3)
                    else:
                        assert metadata.data.numpy.ndim == 2, "Raw saliency map has to be two-dimensional."
                        if raw_sal_map_shape:
                            assert (
                                metadata.data.numpy.shape == raw_sal_map_shape
                            ), "Raw saliency map shape is incorrect."
                    assert metadata.data.numpy.dtype == np.uint8, "Saliency map has to be uint8 dtype."
        if only_predicted:
            assert saliency_map_counter == len(data_point.annotation_scene.get_labels()), assert_text_explain_predicted
        else:
            assert saliency_map_counter == len(task_labels), assert_text_explain_all
