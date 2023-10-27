export CUDA_VISIBLE_DEVICES=0
export REG_RES_ROOT="/tmp"

pytest -v -s ./tests/regression/classification/test_classification.py
pytest -v -s ./tests/regression/detection/test_detection.py
pytest -v -s ./tests/regression/semantic_segmentation/test_segmentation.py
pytest -v -s ./tests/regression/instance_segmentation/test_instance_segmentation.py

python tests/regression/summarize_test_results.py