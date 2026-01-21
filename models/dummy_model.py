#!/usr/bin/env python3
"""
Dummy Model for GENESIS
=======================
A simple dummy model that applies scale and offset to the input signal.
This is useful for testing the diffusion pipeline without training a real model.

Input:  x_sig (B, 2, L) - PMT signals [nPE, firstTime]
        geom (B, 3, L) - PMT geometry [x, y, z] (not used)
        label (B, 6) - Event labels [Energy, Zenith, Azimuth, X, Y, Z] (not used)
        timestep (B,) - Diffusion timestep (not used)

Output: (B, 2, L) - Scaled and offset version of input signal
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Union, List

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DummyModel(nn.Module):
    """
    Dummy model that normalizes input signals using scale and offset.
    
    정규화 공식: output = (input - offset) / scale
    
    각 채널마다 다른 offset과 scale을 적용합니다:
    - 채널 0 (nPE): (x - offset[0]) / scale[0]
    - 채널 1 (firstTime): (x - offset[1]) / scale[1]
    
    Args:
        scale: Scale 값 (채널별로 리스트로 지정 가능, 예: [200.0, 10.0])
        offset: Offset 값 (채널별로 리스트로 지정 가능, 예: [0.0, 0.0])
    """
    
    def __init__(
        self,
        scale: Union[List[float], float] = [1.0, 1.0],
        offset: Union[List[float], float] = [0.0, 0.0],
    ):
        super().__init__()
        
        # scale과 offset을 리스트로 변환
        if isinstance(scale, (int, float)):
            scale = [scale, scale]
        if isinstance(offset, (int, float)):
            offset = [offset, offset]
        
        # Fixed 파라미터로 저장 (학습 안 함)
        # shape: (2, 1, 1) - 채널 2개, 배치와 길이 차원은 브로드캐스팅
        self.register_buffer("scale", torch.tensor(scale).view(2, 1, 1).float())
        self.register_buffer("offset", torch.tensor(offset).view(2, 1, 1).float())
    
    def forward(
        self,
        x_sig: torch.Tensor,
        timestep: torch.Tensor,
        geom: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: 정규화 적용 (x - offset) / scale
        
        Args:
            x_sig: Input signal (B, 2, L) - [nPE, firstTime]
            timestep: Diffusion timestep (B,) - 무시됨
            geom: Geometry (B, 3, L) - 무시됨
            label: Event labels (B, 6) - 무시됨
        
        Returns:
            정규화된 신호 (B, 2, L)
        """
        # 브로드캐스팅을 위해 reshape: (2, 1, 1) -> (1, 2, 1)
        scale = self.scale.view(1, -1, 1)  # (1, 2, 1)
        offset = self.offset.view(1, -1, 1)  # (1, 2, 1)
        
        # 정규화: (x - offset) / scale
        output = (x_sig - offset) / scale
        
        return output


def create_dummy_model(
    scale: Union[List[float], float] = [1.0, 1.0],
    offset: Union[List[float], float] = [0.0, 0.0],
) -> DummyModel:
    """
    Factory function to create a DummyModel instance.
    
    Args:
        scale: Scale 값 (채널별 리스트 또는 단일 값)
        offset: Offset 값 (채널별 리스트 또는 단일 값)
    
    Returns:
        DummyModel instance
    """
    return DummyModel(scale=scale, offset=offset)


if __name__ == "__main__":
    import argparse
    from dataloader.h5 import H5Dataset
    
    parser = argparse.ArgumentParser(description="Test DummyModel with H5 dataset")
    parser.add_argument(
        "-f", "--h5_path",
        type=str,
        default="/home/pmj0324/icecube-genesis/0121/GENESIS/GENESIS-data/22644_0921_time_shift.h5",
        help="Path to H5 file (uses H5Dataset from dataloader/h5.py)",
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=1,
        help="Batch size for testing",
    )
    parser.add_argument(
        "-s", "--scale",
        type=str,
        default="200.0,10.0",
        help="Scale 값 (채널별, 쉼표로 구분, 예: '200.0,10.0')",
    )
    parser.add_argument(
        "-o", "--offset",
        type=str,
        default="0.0,0.0",
        help="Offset 값 (채널별, 쉼표로 구분, 예: '0.0,0.0')",
    )
    
    args = parser.parse_args()
    
    # Parse scale and offset from string
    scale = [float(x.strip()) for x in args.scale.split(",")]
    offset = [float(x.strip()) for x in args.offset.split(",")]
    
    print("=" * 80)
    print("Testing DummyModel with H5 Dataset")
    print("=" * 80)
    
    # Load dataset
    print(f"\nLoading dataset: {args.h5_path}")
    dataset = H5Dataset(h5_path=args.h5_path)
    print(f"Dataset: {len(dataset)} events, waveform length: {dataset.waveform_len}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print(f"Model: scale={scale}, offset={offset}")
    model = DummyModel(scale=scale, offset=offset)
    model.eval()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x_sig = batch["sig"].to(device)
            geom = batch["geo"].to(device)
            label = batch["label"].to(device)
            timestep = torch.randint(0, 1000, (x_sig.shape[0],), device=device)
            
            output = model(x_sig, timestep, geom, label)
            
            # Verify output
            scale_tensor = torch.tensor(scale, device=device).view(1, -1, 1)
            offset_tensor = torch.tensor(offset, device=device).view(1, -1, 1)
            expected = (x_sig - offset_tensor) / scale_tensor
            diff = (output - expected).abs().max()
            
            print(f"Batch {batch_idx + 1}: shape={x_sig.shape}, "
                  f"input_range=[{x_sig.min():.2f}, {x_sig.max():.2f}], "
                  f"output_range=[{output.min():.2f}, {output.max():.2f}], "
                  f"max_diff={diff:.6f}", end="")
            
            if diff < 1e-5:
                print(" [PASS]")
            else:
                print(" [FAIL]")
            
            if batch_idx == 0:
                break
    
    print("\n" + "=" * 80)
    print("Test completed")
    print("=" * 80)