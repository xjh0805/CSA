import evaluation
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RUN_PATH = "/data/"

DATA_PATH = "/data/"
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="test")


