import numpy as np

import torch
import torch.nn as nn

from captum.attr import visualization

def visualize_attributions(visualization_records):
    cast_records = []
    for record in visualization_records:
        # Each record is assumed to be a tuple
        print(record)
        cast_records.append(visualization.VisualizationDataRecord(
            *record
            ))
    visualization.visualize_text(cast_records)
