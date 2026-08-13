"""Remote GPU smoke test for the BF16 FlashAttention 2 training path."""

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA GPU does not support BF16.")

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute capability: sm_{capability[0]}{capability[1]}")
    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")

    torch.manual_seed(42)
    query = torch.randn(
        2,
        512,
        32,
        128,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn(
        2, 512, 8, 128, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    value = torch.randn(
        2, 512, 8, 128, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    flash_output = flash_attn_func(query, key, value, dropout_p=0.0, causal=True)
    flash_output.float().square().mean().backward()

    if not torch.isfinite(flash_output).all():
        raise RuntimeError("FlashAttention produced non-finite output.")
    for tensor_name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise RuntimeError(
                f"FlashAttention produced a missing or non-finite {tensor_name} gradient."
            )

    # Use CPU float32 SDPA as a small independent forward reference. The remote
    # test intentionally exercises head dimension 128, which Mistral 7B uses.
    reference_query = query.detach().float().cpu().transpose(1, 2)
    reference_key = key.detach().float().cpu().repeat_interleave(4, dim=2).transpose(1, 2)
    reference_value = value.detach().float().cpu().repeat_interleave(4, dim=2).transpose(1, 2)
    reference_output = F.scaled_dot_product_attention(
        reference_query,
        reference_key,
        reference_value,
        dropout_p=0.0,
        is_causal=True,
    ).transpose(1, 2)
    error = (flash_output.detach().float().cpu() - reference_output).abs()
    print(f"Forward max absolute error: {error.max().item():.6f}")
    print(f"Forward mean absolute error: {error.mean().item():.6f}")
    if error.max().item() > 0.1:
        raise RuntimeError("FlashAttention differs unexpectedly from float32 SDPA.")

    print("BF16 FlashAttention 2 forward and backward smoke test passed.")


if __name__ == "__main__":
    main()
