"""User-run environment check; does not download anything or alter packages."""
import platform
import sys
from pathlib import Path

import torch

print('Python:', platform.python_version(), sys.executable)
print('PyTorch:', torch.__version__, 'built CUDA:', torch.version.cuda)
assert sys.version_info[:3] == (3, 12, 9), 'Expected Python 3.12.9'
assert sys.prefix != sys.base_prefix, 'Activate the project .venv'
assert torch.cuda.is_available(), 'CUDA unavailable. Check nvidia-smi and PyTorch wheel.'
for i in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {prop.name}; {prop.total_memory / 2**30:.2f} GiB; SM {prop.major}.{prop.minor}')
print('Native BF16:', torch.cuda.is_bf16_supported(including_emulation=False))
print('Training/retrieval precision: FP16 on CUDA; FP32 on CPU.')
print('The unchanged official evaluator uses BF16, which is not native on RTX 2080 Ti.')
print('Check evaluator compatibility on a smoke subset before a full evaluation.')
print('Project:', Path(__file__).resolve().parents[2])
