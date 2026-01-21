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
        "--h5_path",
        type=str,
        default="/home/pmj0324/icecube-genesis/0121/GENESIS/GENESIS-data/22644_0921_time_shift.h5",
        help="Path to H5 file (uses H5Dataset from dataloader/h5.py)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="200.0,10.0",
        help="Scale 값 (채널별, 쉼표로 구분, 예: '200.0,10.0')",
    )
    parser.add_argument(
        "--offset",
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
    
    # Load dataset using H5Dataset from dataloader/h5.py
    print(f"\n📂 Loading dataset using H5Dataset from dataloader/h5.py")
    print(f"   H5 file path: {args.h5_path}")
    dataset = H5Dataset(h5_path=args.h5_path)
    print(f"✅ Dataset loaded: {len(dataset)} events")
    print(f"   Waveform length: {dataset.waveform_len}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 for testing
    )
    
    # Create dummy model
    print(f"\n🤖 Creating DummyModel:")
    print(f"   Scale (채널별): {scale}")
    print(f"   Offset (채널별): {offset}")
    model = DummyModel(scale=scale, offset=offset)
    model.eval()
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    model = model.to(device)
    
    # Test with a few batches
    print(f"\n🔄 Testing forward pass...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            x_sig = batch["sig"].to(device)  # (B, 2, L)
            geom = batch["geo"].to(device)  # (B, 3, L)
            label = batch["label"].to(device)  # (B, 6)
            
            # Create dummy timestep
            timestep = torch.randint(0, 1000, (x_sig.shape[0],), device=device)
            
            print(f"\n  Batch {batch_idx + 1}:")
            print(f"    Signal shape: {x_sig.shape}")
            print(f"    Geometry shape: {geom.shape}")
            print(f"    Label shape: {label.shape}")
            print(f"    Timestep shape: {timestep.shape}")
            print(f"    Signal range: [{x_sig.min():.4f}, {x_sig.max():.4f}]")
            print(f"    Signal mean: {x_sig.mean():.4f}, std: {x_sig.std():.4f}")
            
            # Forward pass
            output = model(x_sig, timestep, geom, label)
            
            print(f"    Output shape: {output.shape}")
            print(f"    Output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"    Output mean: {output.mean():.4f}, std: {output.std():.4f}")
            
            # Verify output = (input - offset) / scale
            scale_tensor = torch.tensor(scale, device=device).view(1, -1, 1)
            offset_tensor = torch.tensor(offset, device=device).view(1, -1, 1)
            expected = (x_sig - offset_tensor) / scale_tensor
            diff = (output - expected).abs().max()
            print(f"    Max difference from expected: {diff:.6f}")
            
            if diff < 1e-5:
                print(f"    ✅ Output matches expected (정규화 공식: (x - offset) / scale)")
            else:
                print(f"    ⚠️  Output differs from expected!")
            
            # Only test first batch
            if batch_idx == 0:
                break
    
    print(f"\n{'=' * 80}")
    print("✅ DummyModel test with H5 dataset completed!")
    print(f"{'=' * 80}")

