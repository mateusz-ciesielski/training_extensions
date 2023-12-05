# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import os.path as osp
import tempfile

import pytest
import torch
import numpy as np

from otx.algorithms.classification.adapters.mmcls.task import MMClassificationTask
from otx.algorithms.classification.adapters.openvino.task import ClassificationOpenVINOTask

from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelEntity,
)
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.train_parameters import TrainParameters
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.cli.utils.io import read_model, save_model_data
from tests.e2e.api.xai.xai_test_helper import saliency_maps_check
from tests.integration.api.classification.test_api_classification import (
    DEFAULT_CLS_TEMPLATE_DIR,
    ClassificationTaskAPIBase,
)

# from tests.test_suite.e2e_test_system import e2e_pytest_api

torch.manual_seed(0)


class TestClsXAIAPI:
    ref_raw_saliency_shapes = {
        "EfficientNet-B0": (7, 7),
    }

    @pytest.mark.parametrize(
        "multilabel,hierarchical",
        [(False, False), (True, False), (False, True)],
        ids=["multiclass", "multilabel", "hierarchical"],
    )
    def test_inference_xai(self, multilabel, hierarchical, tmp_dir_path):
        hyper_parameters, model_template = ClassificationTaskAPIBase.setup_configurable_parameters(
            DEFAULT_CLS_TEMPLATE_DIR, num_iters=1
        )
        task_environment, dataset = ClassificationTaskAPIBase.init_environment(
            hyper_parameters, model_template, multilabel, hierarchical, 20
        )

        # Train and save a model
        task = MMClassificationTask(task_environment=task_environment)
        train_parameters = TrainParameters()
        output_model = ModelEntity(
            dataset,
            task_environment.get_model_configuration(),
        )
        task.train(dataset, output_model, train_parameters)
        save_model_data(output_model, tmp_dir_path)

        for processed_saliency_maps, only_predicted in [[True, False], [False, True]]:
            task_environment, dataset = ClassificationTaskAPIBase.init_environment(
                hyper_parameters, model_template, multilabel, hierarchical, 20
            )

            # Infer torch model
            task = MMClassificationTask(task_environment=task_environment)
            inference_parameters = InferenceParameters(
                is_evaluation=False,
                process_saliency_maps=processed_saliency_maps,
                explain_predicted_classes=only_predicted,
            )
            predicted_dataset = task.infer(dataset.with_empty_annotations(), inference_parameters)

            # Check saliency maps torch task
            task_labels = output_model.configuration.get_label_schema().get_labels(include_empty=False)
            saliency_maps_check(
                predicted_dataset,
                task_labels,
                self.ref_raw_saliency_shapes[model_template.name],
                processed_saliency_maps=processed_saliency_maps,
                only_predicted=only_predicted,
            )

            # Save OV IR model
            task._model_ckpt = osp.join(tmp_dir_path, "weights.pth")
            exported_model = ModelEntity(None, task_environment.get_model_configuration())
            task.export(ExportType.OPENVINO, exported_model, dump_features=True)
            os.makedirs(tmp_dir_path, exist_ok=True)
            save_model_data(exported_model, tmp_dir_path)

            # Infer OV IR model
            load_weights_ov = osp.join(tmp_dir_path, "openvino.xml")
            task_environment.model = read_model(task_environment.get_model_configuration(), load_weights_ov, None)
            task = ClassificationOpenVINOTask(task_environment=task_environment)
            _, dataset = ClassificationTaskAPIBase.init_environment(
                hyper_parameters, model_template, multilabel, hierarchical, 20
            )
            predicted_dataset_ov = task.infer(dataset.with_empty_annotations(), inference_parameters)

            # Check saliency maps OV task
            saliency_maps_check(
                predicted_dataset_ov,
                task_labels,
                self.ref_raw_saliency_shapes[model_template.name],
                processed_saliency_maps=processed_saliency_maps,
                only_predicted=only_predicted,
            )
