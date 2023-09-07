To run a short verion of regression tests, perform the following steps:

- Install OTX following [Quick Start](https://openvinotoolkit.github.io/training_extensions/latest/guide/get_started/quick_start_guide/installation.html#install-openvino-training-extensions-for-users)
- Unpack the test data into `/data/otx/` folder
- Run the following commands:
```bash
pytest -v -s ./tests/regression/classification/test_classification.py
pytest -v -s ./tests/regression/detection/test_detection.py
pytest -v -s ./tests/regression/instance_segmentation/test_instnace_segmentation.py
pytest -v -s ./tests/regression/semantic_segmentation/test_segmentation.py

python tests/regression/summarize_test_results.py --output_path <output folder>

```

The resulting report will be saved to `<output folder>/non_anomaly_results.csv`