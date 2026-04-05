import numpy as np
from typing import List, Dict

WORKLOAD_SPECS = {
    'ResNet-50': [
        {'type': 'conv2d', 'K': 64,  'C': 3,   'R': 7, 'S': 7,  'P': 112, 'Q': 112},
        {'type': 'conv2d', 'K': 64,  'C': 64,  'R': 1, 'S': 1,  'P': 56,  'Q': 56},
        {'type': 'conv2d', 'K': 256, 'C': 64,  'R': 1, 'S': 1,  'P': 56,  'Q': 56},
        {'type': 'conv2d', 'K': 128, 'C': 256, 'R': 1, 'S': 1,  'P': 28,  'Q': 28},
        {'type': 'conv2d', 'K': 512, 'C': 128, 'R': 1, 'S': 1,  'P': 28,  'Q': 28},
    ],
    'BERT-Base': [
        {'type': 'matmul', 'M': 512, 'K': 768, 'N': 768},   # attention
        {'type': 'matmul', 'M': 512, 'K': 768, 'N': 3072},  # FFN up
        {'type': 'matmul', 'M': 512, 'K': 3072, 'N': 768},  # FFN down
    ],
    'MobileNetV2': [
        {'type': 'conv2d', 'K': 32,  'C': 3,   'R': 3, 'S': 3,  'P': 112, 'Q': 112},
        {'type': 'conv2d', 'K': 32,  'C': 32,  'R': 3, 'S': 3,  'P': 112, 'Q': 112}, # depthwise proxy
        {'type': 'conv2d', 'K': 16,  'C': 32,  'R': 1, 'S': 1,  'P': 112, 'Q': 112},
    ],
    'EfficientNet-B4': [
        {'type': 'conv2d', 'K': 48,  'C': 3,   'R': 3, 'S': 3,  'P': 190, 'Q': 190},
        {'type': 'conv2d', 'K': 48,  'C': 48,  'R': 3, 'S': 3,  'P': 190, 'Q': 190}, 
        {'type': 'conv2d', 'K': 24,  'C': 48,  'R': 1, 'S': 1,  'P': 190, 'Q': 190},
    ],
    'ViT-B/16': [
        {'type': 'conv2d', 'K': 768, 'C': 3,   'R': 16, 'S': 16, 'P': 14, 'Q': 14}, # patch embed
        {'type': 'matmul', 'M': 197, 'K': 768, 'N': 768},   # attention 
        {'type': 'matmul', 'M': 197, 'K': 768, 'N': 3072},  # MLP
        {'type': 'matmul', 'M': 197, 'K': 3072, 'N': 768},
    ]
}

WORKLOAD_ALIASES = {
    'ResNet50': 'ResNet-50',
    'MobileNetV2': 'MobileNetV2',
    'EfficientNetB4': 'EfficientNet-B4',
    'BERTBase': 'BERT-Base',
    'ViTB16': 'ViT-B/16',
    'ViT-B-16': 'ViT-B/16',
}

class WorkloadRunner:
    """
    Generates execution patterns and returns layer specs.
    """
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.available_workloads = list(WORKLOAD_SPECS.keys())

    def normalize_workload_name(self, workload_name: str) -> str:
        if workload_name in WORKLOAD_SPECS:
            return workload_name
        return WORKLOAD_ALIASES.get(workload_name, workload_name)
        
    def generate_stream(self, pattern: str, workload_names: List[str], total_steps: int, seed: int) -> List[str]:
        """
        Creates an ordered sequence of runs based on a given scheduler pattern.
        """
        rng = np.random.RandomState(seed)
        valid_names = [w for w in workload_names if w in self.available_workloads]
        if not valid_names:
            valid_names = self.available_workloads
            
        stream = []
        if pattern == 'static':
            stream = [valid_names[0]] * total_steps
        elif pattern == 'alternating':
            for i in range(total_steps):
                stream.append(valid_names[i % len(valid_names)])
        elif pattern == 'mixed':
            indices = rng.randint(0, len(valid_names), size=total_steps)
            stream = [valid_names[idx] for idx in indices]
        elif pattern == 'burst':
            current = valid_names[0]
            for _ in range(total_steps):
                if rng.rand() > 0.9: # 10% chance to switch burst target
                    current = valid_names[rng.randint(0, len(valid_names))]
                stream.append(current)
        else:
            stream = [valid_names[0]] * total_steps
            
        return stream
        
    def get_workload_layers(self, workload_name: str) -> List[dict]:
        """
        Fetches layer parameters for the specified workload.
        """
        canonical_name = self.normalize_workload_name(workload_name)
        if canonical_name in WORKLOAD_SPECS:
            return WORKLOAD_SPECS[canonical_name]
        return WORKLOAD_SPECS['ResNet-50'] # default fallback
