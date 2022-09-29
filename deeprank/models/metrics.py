import lzma
import os
import csv
from typing import List, Tuple, Any, Optional
from math import sqrt

from torch import argmax, tensor
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from deeprank.models.variant import VariantClass
from deeprank.tools.metrics import get_labels_from_output, get_labels_from_targets


class MetricsExporter:
    "The class implements an object, to be called when a neural network generates output"

    def __enter__(self):
        "overridable"
        return self

    def __exit__(self, exception_type, exception, traceback):
        "overridable"
        pass

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "the entry_names, output_values, target_values MUST have the same length"
        pass


class MetricsExporterCollection:
    "allows a series of metrics exporters to be used at the same time"

    def __init__(self, *args: Tuple[MetricsExporter]):
        self._metrics_exporters = args

    def __enter__(self):
        for metrics_exporter in self._metrics_exporters:
            metrics_exporter.__enter__()

        return self

    def __exit__(self, exception_type, exception, traceback):
        for metrics_exporter in self._metrics_exporters:
            metrics_exporter.__exit__(exception_type, exception, traceback)

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        for metrics_exporter in self._metrics_exporters:
            metrics_exporter.process(pass_name, epoch_number, entry_names, output_values, target_values)


class TensorboardVariantClassificationExporter(MetricsExporter):
    "exports to tensorboard, works for variant classification only"

    def __init__(self, directory_path: str, unknown_treshold: Optional[float] = 0.5):
        """
        Args:
            directory_path: where to store tensorboard files
            unknown_treshold: if the absolute difference between the class probabilities is below this value, give the class label UNKNOWN
        """

        self._directory_path = directory_path
        self._writer = SummaryWriter(log_dir=directory_path)
        self._unknown_treshold = unknown_treshold

    def __enter__(self):
        self._writer.__enter__()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self._writer.__exit__(exception_type, exception, traceback)

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "write to tensorboard"

        output_tensor = tensor(output_values)
        target_tensor = tensor(target_values)

        loss = cross_entropy(output_tensor, target_tensor)
        self._writer.add_scalar(f"{pass_name} loss", loss, epoch_number)

        # lists of VariantClass values
        prediction_labels = get_labels_from_output(output_tensor, unknown_treshold=self._unknown_treshold)
        target_labels = get_labels_from_targets(target_tensor)

        roc_probabilities = []  # floating point values
        roc_targets = []  # list of 0/1 values

        fp, fn, tp, tn = 0, 0, 0, 0
        for entry_index, entry_name in enumerate(entry_names):
            prediction_label = prediction_labels[entry_index]
            target_label = target_labels[entry_index]

            if prediction_label == VariantClass.PATHOGENIC and target_label == VariantClass.PATHOGENIC:
                tp += 1

            elif prediction_label == VariantClass.BENIGN and target_label == VariantClass.BENIGN:
                tn += 1

            elif prediction_label == VariantClass.PATHOGENIC and target_label == VariantClass.BENIGN:
                fp += 1

            elif prediction_label == VariantClass.BENIGN and target_label == VariantClass.PATHOGENIC:
                fn += 1

            if prediction_label != VariantClass.UNKNOWN:
                roc_probabilities.append(output_values[entry_index][1])
                roc_targets.append(target_values[entry_index])

            # Furthermore, UNKNOWN variants are completely ignored..

        mcc_numerator = tn * tp - fp * fn
        if mcc_numerator == 0.0:
            self._writer.add_scalar(f"{pass_name} MCC", 0.0, epoch_number)
        else:
            mcc_denominator = sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))

            if mcc_denominator != 0.0:
                mcc = mcc_numerator / mcc_denominator
                self._writer.add_scalar(f"{pass_name} MCC", mcc, epoch_number)

        accuracy_denominator = tp + tn + fp + fn
        if accuracy_denominator > 0:
            accuracy = (tp + tn) / accuracy_denominator
            self._writer.add_scalar(f"{pass_name} accuracy", accuracy, epoch_number)

        # for ROC curves to work, we need both class values in the target set
        if len(set(roc_targets)) >= 2:
            roc_auc = roc_auc_score(roc_targets, roc_probabilities)
            self._writer.add_scalar(f"{pass_name} ROC AUC", roc_auc, epoch_number)


class OutputExporter(MetricsExporter):
    """ A metrics exporter that writes output tables, containing every single data point.
        The files, that this generates, can be used to compute plots after the run.
    """

    def __init__(self, directory_path):
        self._directory_path = directory_path

    def get_filename(self, pass_name, epoch_number):
        "returns the filename for the table"
        return os.path.join(self._directory_path, f"output-{pass_name}-epoch-{epoch_number}.csv.xz")

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "write the output to the table"

        if not os.path.isdir(self._directory_path):
            os.mkdir(self._directory_path)

        with lzma.open(self.get_filename(pass_name, epoch_number), 'wt') as f:
            w = csv.writer(f)

            w.writerow(["entry", "output", "target"])

            for entry_index, entry_name in enumerate(entry_names):
                output_value = output_values[entry_index]
                target_value = target_values[entry_index]

                w.writerow([entry_name, str(output_value), str(target_value)])


class LabelExporter(MetricsExporter):
    "writes all labels to a table"

    def __init__(self, directory_path, unknown_treshold: Optional[float] = 0.5):
        self._directory_path = directory_path
        self._unknown_treshold = unknown_treshold

    def get_filename(self, pass_name, epoch_number):
        "returns the filename for the table"
        return os.path.join(self._directory_path, f"labels-{pass_name}-epoch-{epoch_number}.csv.xz")

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "write the output to the table"

        # lists of VariantClass values
        output_labels = get_labels_from_output(tensor(output_values), unknown_treshold=self._unknown_treshold)
        target_labels = get_labels_from_targets(tensor(target_values))

        if not os.path.isdir(self._directory_path):
            os.mkdir(self._directory_path)

        with lzma.open(self.get_filename(pass_name, epoch_number), 'wt') as f:
            w = csv.writer(f)

            w.writerow(["entry", "output_label", "target_label"])

            for entry_index, entry_name in enumerate(entry_names):
                output_name = output_labels[entry_index].name
                target_name = target_labels[entry_index].name

                w.writerow([entry_name, output_name, target_name])
