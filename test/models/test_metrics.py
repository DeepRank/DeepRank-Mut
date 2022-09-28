import tempfile
import shutil
import os

from deeprank.models.metrics import TensorboardVariantClassificationExporter, OutputExporter, LabelExporter


test_entries = ["entry0", "entry1"]
test_outputs = [[0.0, 0.1], [2.03, -1.0]]
test_targets = [1, 0]


def test_tensorboard_class_output():

    tmp_dir_path = tempfile.mkdtemp()
    try:
        exporter = TensorboardVariantClassificationExporter(tmp_dir_path)

        with exporter:
            exporter.process("unit-testing", 0, test_entries, test_outputs, test_targets)

            assert len(os.listdir(tmp_dir_path)) > 0, "tensorboard directory is empty"
    finally:
        shutil.rmtree(tmp_dir_path)


def test_output():

    tmp_dir_path = tempfile.mkdtemp()
    try:
        exporter = OutputExporter(tmp_dir_path)

        with exporter:
            exporter.process("unit-testing", 0, test_entries, test_outputs, test_targets)

            assert len(os.listdir(tmp_dir_path)) > 0, "output directory is empty"
    finally:
        shutil.rmtree(tmp_dir_path)


def test_label():
    tmp_dir_path = tempfile.mkdtemp()
    try:

        exporter = LabelExporter(tmp_dir_path)

        with exporter:
            exporter.process("unit-testing", 0, test_entries, test_outputs, test_targets)

            assert len(os.listdir(tmp_dir_path)) > 0, "output directory is empty"
    finally:
        shutil.rmtree(tmp_dir_path)
