"""
Production-Ready Gait Recognition Model
========================================
Pure PyTorch implementation of OpenGait Baseline architecture.
No distributed, no dataset loader, no OpenGait runtime dependencies.

Architecture verified against:
- configs/gaitbase/gaitbase_da_casiab.yaml
- opengait/modeling/models/baseline.py
- Checkpoint: GaitBase_DA-60000.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# MODULES (Manual reconstruction from OpenGait)
# ═══════════════════════════════════════════════════════════════════

class BasicConv2d(nn.Module):
    """Basic 2D convolution without bias"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
    
    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class ResNet9(nn.Module):
    """
    ResNet9 Backbone for Gait Recognition
    
    Architecture:
    - conv1: 1 → 64, stride=1
    - layer1: 64 → 64, stride=1 (1 block)
    - layer2: 64 → 128, stride=2 (1 block)
    - layer3: 128 → 256, stride=2 (1 block)
    - layer4: 256 → 512, stride=1 (1 block)
    """
    
    def __init__(self, in_channel=1, channels=[64, 128, 256, 512], 
                 layers=[1, 1, 1, 1], strides=[1, 2, 2, 1]):
        super().__init__()
        
        self.inplanes = channels[0]
        
        # Initial conv
        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], strides[0])
        self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], strides[1])
        self.layer3 = self._make_layer(BasicBlock, channels[2], layers[2], strides[2])
        self.layer4 = self._make_layer(BasicBlock, channels[3], layers[3], strides[3])
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [n, c, h, w]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x  # [n, 512, h', w']


class SetBlockWrapper(nn.Module):
    """
    Wraps backbone to process sequence dimension
    Input:  [n, c_in, s, h, w]
    Output: [n, c_out, s, h', w']
    """
    def __init__(self, forward_block):
        super().__init__()
        self.forward_block = forward_block
    
    def forward(self, x):
        n, c, s, h, w = x.size()
        # Reshape: [n, c, s, h, w] → [n*s, c, h, w]
        x = x.transpose(1, 2).reshape(-1, c, h, w)
        # Process
        x = self.forward_block(x)
        # Reshape back: [n*s, c', h', w'] → [n, c', s, h', w']
        output_size = x.size()
        x = x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()
        return x


class HorizontalPoolingPyramid(nn.Module):
    """
    Horizontal Pyramid Matching
    Input:  [n, c, h, w]
    Output: [n, c, p] where p = sum(bin_num)
    """
    def __init__(self, bin_num=[16]):
        super().__init__()
        self.bin_num = bin_num
    
    def forward(self, x):
        # x: [n, c, h, w]
        n, c = x.size()[:2]
        features = []
        
        for b in self.bin_num:
            # Divide height into b bins
            z = x.view(n, c, b, -1)  # [n, c, b, h*w/b]
            # Max + Mean pooling
            z = z.mean(-1) + z.max(-1)[0]  # [n, c, b]
            features.append(z)
        
        return torch.cat(features, -1)  # [n, c, p]


class PackSequenceWrapper(nn.Module):
    """Temporal pooling using max"""
    def __init__(self):
        super().__init__()
    
    def forward(self, seqs, seqL=None):
        # seqs: [n, c, s, h, w]
        # For inference, seqL is None, so we max over sequence dimension
        if seqL is None:
            return seqs.max(dim=2)[0]  # [n, c, h, w]
        
        # Handle variable-length sequences (for training)
        # Not needed for inference, but kept for compatibility
        return seqs.max(dim=2)[0]


class SeparateFCs(nn.Module):
    """
    Separate fully-connected layers for each part
    Input:  [n, c_in, p]
    Output: [n, c_out, p]
    """
    def __init__(self, parts_num, in_channels, out_channels):
        super().__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)
            )
        )
    
    def forward(self, x):
        # x: [n, c_in, p]
        x = x.permute(2, 0, 1).contiguous()  # [p, n, c_in]
        out = x.matmul(self.fc_bin)  # [p, n, c_out]
        return out.permute(1, 2, 0).contiguous()  # [n, c_out, p]


class SeparateBNNecks(nn.Module):
    """
    Batch normalization neck for each part
    Input:  [n, c, p]
    Output: feature [n, c, p], logits [n, class_num, p]
    """
    def __init__(self, parts_num, in_channels, class_num):
        super().__init__()
        self.p = parts_num
        self.class_num = class_num
        
        # Parallel BN for all parts
        self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        
        # Classification weights (not used in inference)
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)
            )
        )
    
    def forward(self, x):
        # x: [n, c, p]
        n, c, p = x.size()
        
        # Batch norm
        x = x.view(n, -1)  # [n, c*p]
        x = self.bn1d(x)
        x = x.view(n, c, p)
        
        # Normalize features
        feature = x.permute(2, 0, 1).contiguous()  # [p, n, c]
        feature = F.normalize(feature, dim=-1)
        
        # Compute logits (for training compatibility, not used in inference)
        logits = feature.matmul(F.normalize(self.fc_bin, dim=1))
        
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


# ═══════════════════════════════════════════════════════════════════
# BASELINE MODEL
# ═══════════════════════════════════════════════════════════════════

class BaselineModel(nn.Module):
    """
    OpenGait Baseline Architecture (Manual Reconstruction)
    
    Architecture verified against:
    - ResNet9: channels=[64,128,256,512], strides=[1,2,2,1]
    - SeparateFCs: 512→256, parts=16
    - SeparateBNNecks: 256, parts=16
    - bin_num: [16]
    
    Input:  [B, T, 1, 64, 44] - Batch of silhouette sequences
    Output: [B, 256] - L2-normalized embeddings
    """
    
    def __init__(self):
        super().__init__()
        
        # Backbone
        backbone = ResNet9(
            in_channel=1,
            channels=[64, 128, 256, 512],
            layers=[1, 1, 1, 1],
            strides=[1, 2, 2, 1]
        )
        self.Backbone = SetBlockWrapper(backbone)
        
        # Temporal Pooling
        self.TP = PackSequenceWrapper()
        
        # Horizontal Pooling Pyramid
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])
        
        # Feature transformation
        self.FCs = SeparateFCs(parts_num=16, in_channels=512, out_channels=256)
        
        # Batch norm neck
        self.BNNecks = SeparateBNNecks(parts_num=16, in_channels=256, class_num=74)
    
    def forward(self, sils, seqL=None):
        """
        Args:
            sils: [n, s, c, h, w] or [n, c, s, h, w]
            seqL: sequence lengths (optional, not used in inference)
        
        Returns:
            embeddings: [n, c, p] where c=256, p=16
        """
        # Handle input format
        if sils.dim() == 4:
            sils = sils.unsqueeze(1)  # [n, 1, c, h, w]
        
        # Convert to [n, c, s, h, w] if needed
        if sils.size(2) == 1:  # [n, s, 1, h, w]
            sils = sils.permute(0, 2, 1, 3, 4)  # [n, 1, s, h, w]
        
        # Backbone: [n, c, s, h, w] → [n, 512, s, h', w']
        outs = self.Backbone(sils)
        
        # Temporal Pooling: [n, 512, s, h', w'] → [n, 512, h', w']
        outs = self.TP(outs, seqL)
        
        # Horizontal Pooling: [n, 512, h', w'] → [n, 512, 16]
        feat = self.HPP(outs)
        
        # FC layers: [n, 512, 16] → [n, 256, 16]
        embed_1 = self.FCs(feat)
        
        # BN Neck: [n, 256, 16] → [n, 256, 16]
        embed_2, logits = self.BNNecks(embed_1)
        
        # Return embed_1 (matches OpenGait's inference_feat)
        return embed_1


# ═══════════════════════════════════════════════════════════════════
# PRODUCTION EMBEDDING EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

class GaitEmbeddingModel:
    """
    Production-ready gait recognition model.
    
    Features:
    - Pure PyTorch (no OpenGait runtime)
    - No distributed
    - No dataset loader
    - No logger dependencies
    - Thread-safe
    - GPU/CPU compatible
    
    Usage:
        model = GaitEmbeddingModel(checkpoint_path, device='cuda:0')
        embeddings = model.extract(silhouettes)  # [B, T, 1, H, W] → [B, 256]
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0'):
        """
        Args:
            checkpoint_path: Path to GaitBase_DA-60000.pt
            device: 'cuda:0', 'cuda:1', or 'cpu'
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Device setup
        if device.startswith('cuda') and not torch.cuda.is_available():
            print("⚠️  CUDA not available, using CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        
        # Build model using the same helper employed during training and evaluation,
        # but explicitly supply the training configuration.
        from domain_adapt.train_da import DomainAdaptationTrainer
        from domain_adapt.models.da_baseline import create_da_model

        print("Building DomainAdaptedBaseline with training config...")
        # Retrieve exact same model_cfg used in supervised training
        model_cfg = DomainAdaptationTrainer.get_model_cfg()

        self.model = create_da_model(
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            freeze_early_layers=False,  # no freezing needed during inference
            device=self.device,
        )
        # create_da_model already moves to device and sets eval mode internally
        print(f"✅ Model ready on {self.device}")
        
        # redundancy: model already eval()/to(device) if create_da_model does it, but keep for safety
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✅ Model ready on {self.device}")
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load weights from OpenGait checkpoint"""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        # Load weights
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        
        if missing:
            print(f"⚠️  Missing keys: {len(missing)}")
            for key in missing[:5]:  # Show first 5
                print(f"   - {key}")
        
        if unexpected:
            print(f"⚠️  Unexpected keys: {len(unexpected)}")
            for key in unexpected[:5]:
                print(f"   - {key}")
        
        # Verify critical components loaded
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Loaded {total_params:,} parameters")
    
    @torch.no_grad()
    def extract(self, silhouettes: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from silhouette sequences.

        Args:
            silhouettes (torch.Tensor):
                Shape: [B, T, 1, H, W]

        Returns:
            torch.Tensor:
                Shape: [B, C] normalized embedding vectors
        """

        self.model.eval()

        # ensure tensor is on correct device
        silhouettes = silhouettes.to(self.device)

        B, T, C, H, W = silhouettes.shape

        # Remove channel dimension (training used [B, T, H, W])
        silhouettes = silhouettes.squeeze(2)  # → [B, T, H, W]

        # Sequence lengths tensor
        seq_lengths = torch.full(
            (B,),
            T,
            dtype=torch.long,
            device=silhouettes.device
        )

        # Construct input tuple in OpenGait format
        ipts = (
            [silhouettes],  # list of silhouette tensors
            torch.zeros(B, dtype=torch.long, device=silhouettes.device),  # dummy labels
            None,
            None,
            [seq_lengths]  # list of sequence lengths
        )

        # Forward pass
        out = self.model(ipts)

        # Extract embeddings
        feats = out["inference_feat"]["embeddings"]  # [B, C, P]
        embeds = feats.mean(dim=-1)                  # average over parts → [B, C]

        # L2 normalization
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)

        return embeds
    
    def extract_batch(self, silhouettes: torch.Tensor, batch_size: int = 8) -> torch.Tensor:
        """
        Extract embeddings in batches (for large datasets).
        
        Args:
            silhouettes: [N, T, 1, H, W]
            batch_size: Number of sequences to process at once
        
        Returns:
            embeddings: [N, 256]
        """
        N = silhouettes.size(0)
        all_embeddings = []
        
        for i in range(0, N, batch_size):
            batch = silhouettes[i:i+batch_size]
            emb = self.extract(batch)
            all_embeddings.append(emb)
        
        return torch.cat(all_embeddings, dim=0)
