import numpy as np
from typing import List, Dict

WORKLOAD_SPECS = {
    'ResNet-50': [
        # Stage 1
        {'type': 'conv2d', 'K': 64,  'C': 3,    'R': 7, 'S': 7, 'P': 112, 'Q': 112},
        # Stage 2 (3 blocks)
        {'type': 'conv2d', 'K': 64,  'C': 64,   'R': 3, 'S': 3, 'P': 56,  'Q': 56},
        {'type': 'conv2d', 'K': 64,  'C': 64,   'R': 3, 'S': 3, 'P': 56,  'Q': 56},
        {'type': 'conv2d', 'K': 256, 'C': 64,   'R': 1, 'S': 1, 'P': 56,  'Q': 56},
        # Stage 3 (4 blocks)
        {'type': 'conv2d', 'K': 128, 'C': 256,  'R': 1, 'S': 1, 'P': 28,  'Q': 28},
        {'type': 'conv2d', 'K': 128, 'C': 128,  'R': 3, 'S': 3, 'P': 28,  'Q': 28},
        {'type': 'conv2d', 'K': 512, 'C': 128,  'R': 1, 'S': 1, 'P': 28,  'Q': 28},
        {'type': 'conv2d', 'K': 128, 'C': 512,  'R': 1, 'S': 1, 'P': 28,  'Q': 28},
        # Stage 4 (6 blocks)
        {'type': 'conv2d', 'K': 256, 'C': 512,  'R': 1, 'S': 1, 'P': 14,  'Q': 14},
        {'type': 'conv2d', 'K': 256, 'C': 256,  'R': 3, 'S': 3, 'P': 14,  'Q': 14},
        {'type': 'conv2d', 'K': 1024,'C': 256,  'R': 1, 'S': 1, 'P': 14,  'Q': 14},
        {'type': 'conv2d', 'K': 256, 'C': 1024, 'R': 1, 'S': 1, 'P': 14,  'Q': 14},
        # Stage 5 (3 blocks)
        {'type': 'conv2d', 'K': 512, 'C': 1024, 'R': 1, 'S': 1, 'P': 7,   'Q': 7},
        {'type': 'conv2d', 'K': 512, 'C': 512,  'R': 3, 'S': 3, 'P': 7,   'Q': 7},
        {'type': 'conv2d', 'K': 2048,'C': 512,  'R': 1, 'S': 1, 'P': 7,   'Q': 7},
    ],
    'BERT-Base': [
        # 12 transformer blocks, sampled as 4 representative blocks
        {'type': 'matmul', 'M': 512, 'K': 768,  'N': 768},   # block 1 attn
        {'type': 'matmul', 'M': 512, 'K': 768,  'N': 3072},  # block 1 FFN up
        {'type': 'matmul', 'M': 512, 'K': 3072, 'N': 768},   # block 1 FFN down
        {'type': 'matmul', 'M': 512, 'K': 768,  'N': 768},   # block 2
        {'type': 'matmul', 'M': 512, 'K': 768,  'N': 3072},
        {'type': 'matmul', 'M': 512, 'K': 3072, 'N': 768},
        {'type': 'matmul', 'M': 512, 'K': 768,  'N': 768},   # block 3
        {'type': 'matmul', 'M': 512, 'K': 768,  'N': 3072},
        {'type': 'matmul', 'M': 512, 'K': 3072, 'N': 768},
        {'type': 'matmul', 'M': 512, 'K': 768,  'N': 768},   # block 4
        {'type': 'matmul', 'M': 512, 'K': 768,  'N': 3072},
        {'type': 'matmul', 'M': 512, 'K': 3072, 'N': 768},
    ],
    'MobileNetV2': [
        {'type': 'conv2d', 'K': 32,  'C': 3,   'R': 3, 'S': 3, 'P': 112, 'Q': 112},
        {'type': 'conv2d', 'K': 96,  'C': 32,  'R': 1, 'S': 1, 'P': 112, 'Q': 112},
        {'type': 'conv2d', 'K': 96,  'C': 96,  'R': 3, 'S': 3, 'P': 56,  'Q': 56},
        {'type': 'conv2d', 'K': 24,  'C': 96,  'R': 1, 'S': 1, 'P': 56,  'Q': 56},
        {'type': 'conv2d', 'K': 144, 'C': 24,  'R': 1, 'S': 1, 'P': 56,  'Q': 56},
        {'type': 'conv2d', 'K': 144, 'C': 144, 'R': 3, 'S': 3, 'P': 28,  'Q': 28},
        {'type': 'conv2d', 'K': 32,  'C': 144, 'R': 1, 'S': 1, 'P': 28,  'Q': 28},
        {'type': 'conv2d', 'K': 192, 'C': 32,  'R': 1, 'S': 1, 'P': 28,  'Q': 28},
        {'type': 'conv2d', 'K': 192, 'C': 192, 'R': 3, 'S': 3, 'P': 14,  'Q': 14},
        {'type': 'conv2d', 'K': 64,  'C': 192, 'R': 1, 'S': 1, 'P': 14,  'Q': 14},
        {'type': 'conv2d', 'K': 384, 'C': 64,  'R': 1, 'S': 1, 'P': 14,  'Q': 14},
        {'type': 'conv2d', 'K': 384, 'C': 384, 'R': 3, 'S': 3, 'P': 7,   'Q': 7},
        {'type': 'conv2d', 'K': 96,  'C': 384, 'R': 1, 'S': 1, 'P': 7,   'Q': 7},
    ],
    'EfficientNet-B4': [
        {'type': 'conv2d', 'K': 48,  'C': 3,   'R': 3, 'S': 3, 'P': 190, 'Q': 190},
        {'type': 'conv2d', 'K': 48,  'C': 48,  'R': 3, 'S': 3, 'P': 190, 'Q': 190},
        {'type': 'conv2d', 'K': 24,  'C': 48,  'R': 1, 'S': 1, 'P': 190, 'Q': 190},
        {'type': 'conv2d', 'K': 144, 'C': 24,  'R': 1, 'S': 1, 'P': 95,  'Q': 95},
        {'type': 'conv2d', 'K': 144, 'C': 144, 'R': 3, 'S': 3, 'P': 95,  'Q': 95},
        {'type': 'conv2d', 'K': 32,  'C': 144, 'R': 1, 'S': 1, 'P': 95,  'Q': 95},
        {'type': 'conv2d', 'K': 192, 'C': 32,  'R': 1, 'S': 1, 'P': 48,  'Q': 48},
        {'type': 'conv2d', 'K': 192, 'C': 192, 'R': 5, 'S': 5, 'P': 48,  'Q': 48},
        {'type': 'conv2d', 'K': 56,  'C': 192, 'R': 1, 'S': 1, 'P': 48,  'Q': 48},
        {'type': 'conv2d', 'K': 336, 'C': 56,  'R': 1, 'S': 1, 'P': 24,  'Q': 24},
        {'type': 'conv2d', 'K': 336, 'C': 336, 'R': 5, 'S': 5, 'P': 24,  'Q': 24},
        {'type': 'conv2d', 'K': 112, 'C': 336, 'R': 1, 'S': 1, 'P': 24,  'Q': 24},
        {'type': 'conv2d', 'K': 672, 'C': 112, 'R': 1, 'S': 1, 'P': 12,  'Q': 12},
        {'type': 'conv2d', 'K': 672, 'C': 672, 'R': 3, 'S': 3, 'P': 12,  'Q': 12},
        {'type': 'conv2d', 'K': 160, 'C': 672, 'R': 1, 'S': 1, 'P': 12,  'Q': 12},
    ],
    'ViT-B/16': [
        {'type': 'conv2d', 'K': 768, 'C': 3,    'R': 16, 'S': 16, 'P': 14, 'Q': 14},
        {'type': 'matmul', 'M': 197, 'K': 768,  'N': 768},   # block 1 attn
        {'type': 'matmul', 'M': 197, 'K': 768,  'N': 3072},  # block 1 MLP up
        {'type': 'matmul', 'M': 197, 'K': 3072, 'N': 768},
        {'type': 'matmul', 'M': 197, 'K': 768,  'N': 768},   # block 2
        {'type': 'matmul', 'M': 197, 'K': 768,  'N': 3072},
        {'type': 'matmul', 'M': 197, 'K': 3072, 'N': 768},
        {'type': 'matmul', 'M': 197, 'K': 768,  'N': 768},   # block 3
        {'type': 'matmul', 'M': 197, 'K': 768,  'N': 3072},
        {'type': 'matmul', 'M': 197, 'K': 3072, 'N': 768},
        {'type': 'matmul', 'M': 197, 'K': 768,  'N': 768},   # block 4
        {'type': 'matmul', 'M': 197, 'K': 768,  'N': 3072},
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
