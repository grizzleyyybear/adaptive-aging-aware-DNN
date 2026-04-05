"""Hardware graph and dataset generation."""
from .accelerator_graph import AcceleratorGraph
from .graph_dataset import AcceleratorGraphDataset as AgingDataset

__all__ = ['AcceleratorGraph', 'AgingDataset']
