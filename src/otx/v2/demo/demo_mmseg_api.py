from otx.v2.adapters.torch.mmengine.mmseg import Engine, get_model, Dataset, list_models
from tests.v2.integration.test_helper import TASK_CONFIGURATION
from hydra import initialize_config_dir
from pathlib import Path
from otx.v2.api.utils.importing import get_otx_root_path

CONFIG_PATH = Path(get_otx_root_path()) / "v2/configs/mmseg"

initialize_config_dir(config_dir=str(CONFIG_PATH))

dataset = Dataset(
    train_data_roots=TASK_CONFIGURATION["segmentation"]["train_data_roots"],
    val_data_roots=TASK_CONFIGURATION["segmentation"]["val_data_roots"],
    test_data_roots=TASK_CONFIGURATION["segmentation"]["test_data_roots"],
)

models = list_models()
model = get_model(models[0], num_classes=dataset.num_classes)

engine = Engine(work_dir="./otx-workspace")

# Train (1 epochs)
results = engine.train(
    model=model,
    train_dataloader=dataset.train_dataloader(),
    val_dataloader=dataset.val_dataloader(),
    max_epochs=1,
)

# Validation
val_score = engine.validate()

# Test
test_score = engine.test(test_dataloader=dataset.test_dataloader())

# Prediction with single image
pred_result = engine.predict(
    model=results["model"],
    checkpoint=results["checkpoint"],
    img=TASK_CONFIGURATION["classification"]["sample"],
)
