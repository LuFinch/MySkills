# PyTorch XPU Common Debug Patterns

This file collects common debugging patterns, tips, and accumulated knowledge for debugging PyTorch XPU issues. Add new patterns here as they are discovered.

## SDPA (Scaled Dot-Product Attention)

### Check which SDPA backend is selected
```python
import torch
q = torch.randn(1, 4, 10, 3, device='xpu', dtype=torch.float16)
k = torch.randn(1, 4, 10, 3, device='xpu', dtype=torch.float16)
v = torch.randn(1, 4, 10, 3, device='xpu', dtype=torch.float16)
backend = torch._fused_sdp_choice(q, k, v, None, 0.0, False)
print(f"Backend: {backend}")  # 1=math, 2=flash, 3=mem_efficient, 4=cudnn, 5=overrideable
```

### XPU SDPA Backend Priority Order

1. **Overrideable** (oneDNN `micro_sdpa`) — default for no-grad cases
2. **Flash Attention** — XPU flash attention implementation
3. **Math** — fallback, always available
4. **Efficient Attention** — not supported on XPU (falls back to math)
5. **cuDNN** — not supported on XPU
