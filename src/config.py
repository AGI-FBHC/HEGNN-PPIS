
import os
from pathlib import Path


__all__ = ['Config']

SOURCE_DIR = Path(__file__).resolve().parent


class Config():
    seed = 2020
    MAP_CUTOFF = 14
    DIST_NORM = 15
    alpha = 0.7
    LAMBDA = 1.5

    learning_rate = 1E-3
    weight_decay = 0
    batch_size = 1
    num_workers = 4
    num_classes = 2  # [not bind, bind]
    epochs = 30
    layers = 4
    feature_fusion_alpha = 0.1

    feature_path = str(SOURCE_DIR / "Feature") + os.sep
    graph_path = str(SOURCE_DIR / "Graph") + os.sep
    center = 'SC/'
    Test60_psepos_path = str(SOURCE_DIR / "Feature/psepos/Test60_psepos_SC.pkl")
    Test315_28_psepos_path = str(SOURCE_DIR / "Feature/psepos/Test315-28_psepos_SC.pkl")
    Btest31_psepos_path = str(SOURCE_DIR / "Feature/psepos/Test60_psepos_SC.pkl")
    UBtest31_28_psepos_path = str(SOURCE_DIR / "Feature/psepos/UBtest31-6_psepos_SC.pkl")
    Train335_psepos_path = str(SOURCE_DIR / "Feature/psepos/Train335_psepos_SC.pkl")
    dataset_path = str(SOURCE_DIR / "Dataset") + os.sep
    hypernodes = 3

    test_type = 1  # change test dataset type, 1 -> Test_60, 2 -> Test_315-28, 3 -> BTest_31-6, 4 -> UBtest_31-6
