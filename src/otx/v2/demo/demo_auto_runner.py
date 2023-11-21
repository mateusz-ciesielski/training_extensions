from otx.v2.api.core import AutoRunner

output_dir = "./otx-workspace/OTX-API-test"
task = "segmentation"

auto_runner = AutoRunner(
    work_dir=output_dir,
    task=task,
    train_data_roots="/home/yuchunli/_DATASET/kvasir_36/train_set",
    val_data_roots="/home/yuchunli/_DATASET/kvasir_36/val_set",
    test_data_roots="/home/yuchunli/_DATASET/kvasir_36/val_set",
)

results = auto_runner.train(
    max_epochs=10,
)
assert "model" in results
assert "checkpoint" in results
assert isinstance(results["checkpoint"], str)

# Validation
auto_runner.validate()

# Test
auto_runner.test()

# Prediction with single image
result = auto_runner.predict(
    model=results["model"],
    checkpoint=results["checkpoint"],
    img="/home/yuchunli/_DATASET/kvasir_36/val_set/images/cju7dubap2g0w0801fgl42mg9.png",
)

# Export Openvino IR Model
export_output = auto_runner.export(
    checkpoint=results["checkpoint"],
)
