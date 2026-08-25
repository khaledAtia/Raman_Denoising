"""
ABLATION VARIANT of Pmodel.py  --  gate input mode ("what the hint is made of").

Answers reviewer comment R3.1: the reviewer objects that E_l and B_l "are from different
branches and are not in the same representational space", so the subtraction cannot inherit
the physical meaning of S = X - B. Our own diagnostics (diagnose_hint.py) confirm the
premise: per-channel correlation between E_l and B_l is ~0.01 at the deep and mid stages.

This file isolates whether the SUBTRACTION specifically does any work, or whether the gate
merely needs access to both tensors. Selected by `hint_mode`:

    'subtract'      hint = E_l - B_l              <- reproduces Pmodel.py exactly (default)
    'concat'        hint = [E_l || B_l]           <- gate sees both, no subtraction imposed
    'encoder-only'  hint = E_l                    <- baseline branch does not inform gating

In 'encoder-only' the baseline decoder still runs and still produces the baseline output;
only its contribution to the GATE is removed. That is the correct ablation -- it isolates
whether the baseline informs gating, not whether the baseline head exists.

Only two things differ from Pmodel.py, both marked "# ABLATION:":
    - gate construction   (~line 449)  hint_channels doubled in 'concat' mode
    - AUSequentialUNet._hint / forward (~line 538)

INTERPRETING THE RESULT
    subtract == concat  -> the subtraction is a convenience, not a mechanism. Say so, and
                           describe the hint as learned gating rather than a residual.
    subtract >  concat  -> strong evidence FOR subtraction, since 'concat' has MORE
                           parameters (doubled input width on project_gate and
                           process_evidence) and still loses.
    concat   >  subtract-> ambiguous; 'concat' has more parameters, so a win could be
                           capacity. Report the parameter counts alongside.
    encoder-only ~ subtract -> the baseline branch contributes nothing to gating, which
                           would undercut the cross-branch claim entirely.

Train with:  train_ablation.py --arch rk4 --gate {subtract,concat,encoder-only}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



class HighResPhysicsGate(nn.Module):
    """
    Attention Gate that upsamples deep semantic features FIRST.
    Features a toggle between Concatenation (Dense-style) and Summation (Attention U-Net style).
    Accepts dynamic bias initialization to control the 'default state' of the gate.
    """
    def __init__(self, current_channels, deep_channels, hint_channels, fusion_mode='concat', kernel_size=3, temperature=1.0, bias_value=None):
        super(HighResPhysicsGate, self).__init__()

        self.temperature = temperature
        self.fusion_mode = fusion_mode.lower()
        
        # --- THE REFINED BIAS LOGIC ---
        # If bias_value is provided (even 0.0), we turn the learnable bias ON.
        # If bias_value is None, we completely remove the bias parameter.
        use_bias = bias_value is not None

        # ==========================================
        # 1. INTENT PREPARATION (The Fusion Switch)
        # ==========================================
        if self.fusion_mode == 'concat':
            # CONCAT MODE: 1 Conv layer, high parameter freedom
            self.project_gate = nn.Conv1d(deep_channels + hint_channels, 1, kernel_size=1, bias=use_bias)
            if use_bias:
                nn.init.constant_(self.project_gate.bias, bias_value)
            
        elif self.fusion_mode == 'sum':
            # SUM MODE: Align dimensions -> Sum -> ReLU -> Project to 1
            self.W_g = nn.Conv1d(deep_channels, hint_channels, kernel_size=1, bias=False)
            self.W_x = nn.Conv1d(hint_channels, hint_channels, kernel_size=1, bias=False)
            
            self.psi = nn.Conv1d(hint_channels, 1, kernel_size=1, bias=use_bias)
            if use_bias:
                nn.init.constant_(self.psi.bias, bias_value)
                
            self.relu = nn.ReLU(inplace=True)
            
        else:
            raise ValueError("fusion_mode must be either 'concat' or 'sum'")
        
        # ==========================================
        # 2. EVIDENCE PROCESSING & FUSION
        # ==========================================
        self.process_evidence = nn.Conv1d(hint_channels, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.out_conv = nn.Conv1d(current_channels + 1, current_channels, kernel_size=1)

    def forward(self, current_signal, deep_signal, current_hint):
        target_len = current_signal.shape[-1]

        # --- A. PREPARE INTENT (High-Resolution Fusion) ---
        deep_up = F.interpolate(deep_signal, size=target_len, mode='linear', align_corners=False)
        
        # --- THE FUSION TOGGLE ---
        if self.fusion_mode == 'concat':
            gate_input = torch.cat([deep_up, current_hint], dim=1)
            gate_map = self.project_gate(gate_input)
            
        elif self.fusion_mode == 'sum':
            g_proj = self.W_g(deep_up)
            x_proj = self.W_x(current_hint)
            # Add element-wise, apply non-linearity, project to 1 channel map
            gate_map = self.psi(self.relu(g_proj + x_proj))
            
        # Create the 0.0 to 1.0 mask
        gate = self.sigmoid(gate_map * self.temperature)    
        
        # --- B. PREPARE EVIDENCE ---
        evidence = self.process_evidence(current_hint)
        
        # --- C. CROSS-GATING ---
        confirmed_evidence = evidence * gate
        
        # --- D. FUSION ---
        combined = torch.cat([current_signal, confirmed_evidence], dim=1)
        return self.out_conv(combined)

class SpatialSquelchGate(nn.Module):
    """
    Spatial Attention Gate.
    Looks at the local spatial features and generates a dynamic 1D mask (0.0 to 1.0)
    across the entire sequence length, squelching empty spaces between peaks.
    """
    def __init__(self, in_channels=64, kernel_size=7, temperature=1.0, bias_value=None):
        super(SpatialSquelchGate, self).__init__()
        
        self.temperature = temperature
        use_bias = bias_value is not None
        
        # 1. The Convolution (Separated from Sequential)
        self.spatial_conv = nn.Conv1d(in_channels, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=use_bias)
        
        # Initialize the negative bias to keep the gate closed by default
        if use_bias:
            nn.init.constant_(self.spatial_conv.bias, bias_value)
            
        # 2. THE FIX: Explicitly define the sigmoid so the visualizer can hook it!
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, raw_prediction):
        """
        features: The final 64-channel decoder output (B, 64, L)
        raw_prediction: The 1-channel physical signal prediction (B, 1, L)
        """
        # Compress the 64 feature maps down to a 1-channel spatial map
        gate_map = self.spatial_conv(features)
        
        # Apply temperature and the explicitly defined Sigmoid
        spatial_mask = self.sigmoid(gate_map * self.temperature)
        
        # Multiply the prediction pixel-by-pixel
        return raw_prediction * spatial_mask



class GlobalSquelchSEGate(nn.Module):
    """
    Global Squeeze-and-Excitation Gate.
    Looks at the entire spatial sequence to determine if ANY physical signal exists.
    If it determines the spectrum is blank, it squelches the entire prediction to 0.0.
    """
    def __init__(self, in_channels=64, reduction=4):
        super(GlobalSquelchSEGate, self).__init__()
        
        # 1. "SQUEEZE" - Global Context Gathering
        # We use MaxPool instead of AvgPool because Raman peaks are sparse. 
        # We want to hunt for the existence of ANY sharp activation across the entire length.
        self.squeeze = nn.AdaptiveMaxPool1d(1)
        
        # 2. "EXCITATION" - The Global Decision
        # Compresses the 64 feature channels down to a single probability (0.0 to 1.0)
        self.excitation = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, features, raw_prediction):
        """
        features: The final 64-channel decoder output (B, 64, L)
        raw_prediction: The 1-channel physical signal prediction (B, 1, L)
        """
        # Squeeze: (B, 64, L) -> (B, 64, 1)
        # Summarizes the maximum activation of every filter across the whole spectrum
        global_context = self.squeeze(features)
        
        # Excite: (B, 64, 1) -> (B, 1, 1)
        # Outputs ~1.0 if the network thinks a signal exists, or ~0.0 if it's a blank
        gate_weight = self.excitation(global_context)
        
        # Scale: Broadcast multiply the 1D probability across the entire spatial sequence
        return raw_prediction * gate_weight

class SquelchGate(nn.Module):
    def __init__(self, in_channels,init_val=0.0):
        super(SquelchGate, self).__init__()
        # A small convolution to look at the features and decide "Open/Close"
        self.gate_conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        
        # Learnable Scale Factor (Initialized to 10.0 for sharp transitions)
        self.scale = nn.Parameter(torch.tensor(10.0))
        
        # CRITICAL FIX: Initialize Bias to Negative!
        # This means the gate starts "Closed" (output ~0).
        # The network must push hard (high activation) to open it.
        nn.init.constant_(self.gate_conv.bias, init_val) 

    def forward(self, x, raw_output):
        # x: High-level features from the last decoder layer (Rich context)
        # raw_output: The raw prediction from out_sig (The noisy signal)
        
        # 1. Calculate Gate Value (0.0 to 1.0)
        gate_logits = self.gate_conv(x)
        gate = torch.sigmoid(gate_logits * self.scale)
        
        # 2. Apply Squelch
        return raw_output * gate





class RK4SmoothedBlock(nn.Module):
    """
    Runge-Kutta 4th Order (RK4) Residual Block.
    Maintains the pre-activation and smoothing principles of ResNet-V2,
    but replaces the standard Euler update with an RK4 integration scheme.
    """
    def __init__(self, in_channels, out_channels, smooth_skip=False, pool_kernel=3, step_size=1.0):
        super().__init__()
        
        assert pool_kernel % 2 != 0, "pool_kernel must be an odd number (e.g., 1, 3, 5)"
        self.step_size = step_size
        
        # --- Dynamic Group Allocation ---
        groups_out = 8 if out_channels >= 8 and out_channels % 8 == 0 else 1
        
        # --- State Space Projection (The Shortcut) ---
        # RK4 requires the state 'x' and the function output 'F(x)' to have the 
        # exact same dimensions to compute intermediate k-steps. 
        if in_channels != out_channels:
            # Project the raw input immediately to the output dimension
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            if smooth_skip and pool_kernel > 1:
                dynamic_pad = pool_kernel // 2
                self.shortcut = nn.AvgPool1d(kernel_size=pool_kernel, stride=1, padding=dynamic_pad)
            else:
                self.shortcut = nn.Identity()

        # --- The ODE Function F(x) ---
        # Notice this strictly maps out_channels -> out_channels. 
        # The projection happens in the shortcut BEFORE this block is evaluated.
        self.conv_block = nn.Sequential(
            nn.GroupNorm(num_groups=groups_out, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            
            nn.GroupNorm(num_groups=groups_out, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        # 1. Project input to the unified state space (z_t)
        z = self.shortcut(x)
        h = self.step_size
        
        # 2. Compute the 4 Runge-Kutta intermediate steps
        # k1 = F(z)
        k1 = self.conv_block(z)
        
        # k2 = F(z + (h/2) * k1)
        k2 = self.conv_block(z + (h / 2.0) * k1)
        
        # k3 = F(z + (h/2) * k2)
        k3 = self.conv_block(z + (h / 2.0) * k2)
        
        # k4 = F(z + h * k3)
        k4 = self.conv_block(z + h * k3)
        
        # 3. Final Integration Update (z_{t+1})
        # The skip connection incorporates the weighted average of trajectories
        out = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return out





class SmoothedResBlock(nn.Module):
    """
    Pre-Activation Residual Block (ResNet-V2).
    Moves GroupNorm and ReLU INSIDE the residual path to keep the 
    skip connection mathematically pure and unobstructed.
    """
    def __init__(self, in_channels, out_channels, smooth_skip=False, pool_kernel=3):
        super().__init__()
        
        assert pool_kernel % 2 != 0, "pool_kernel must be an odd number (e.g., 1, 3, 5)"
        
        # --- THE FIX: Dynamic Group Allocation ---
        # If the block is at the very beginning of the network (in_channels=1), 
        # it uses 1 group. Otherwise, it uses the standard 8 groups.
        groups_in = 8 if in_channels >= 8 and in_channels % 8 == 0 else 1
        groups_out = 8 if out_channels >= 8 and out_channels % 8 == 0 else 1
        
        # --- Pre-Activation Residual Path ---
        self.conv_block = nn.Sequential(
            nn.GroupNorm(num_groups=groups_in, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            
            nn.GroupNorm(num_groups=groups_out, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        
        # --- Skip Connection ---
        if in_channels != out_channels:
            # If dimensions change, we project the RAW input. 
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            if smooth_skip and pool_kernel > 1:
                dynamic_pad = pool_kernel // 2
                self.shortcut = nn.AvgPool1d(kernel_size=pool_kernel, stride=1, padding=dynamic_pad)
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        # The skip connection stays purely linear, ensuring a perfect gradient highway.
        return self.shortcut(x) + self.conv_block(x)

# --- 2. DOWNSAMPLING ---
class Down(nn.Module):
    """Downscaling with Strided Convolution"""
    def __init__(self, in_channels, out_channels, smooth_skip=False, pool_kernel=3):
        super().__init__()
        
        # Upgraded to keep GroupNorm consistent across the entire network
        groups = 8
        
        self.down = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=in_channels), 
            nn.ReLU(inplace=True),
            
            # Pass the smoothing parameters down to the ResBlock
            RK4SmoothedBlock(in_channels, out_channels, smooth_skip=smooth_skip, pool_kernel=pool_kernel)
        )

    def forward(self, x):
        return self.down(x)


# --- 3. UPSAMPLING ---
class Up(nn.Module):
    """Upscaling then SmoothedResBlock"""
    def __init__(self, in_channels, out_channels, bilinear=False, smooth_skip=False, pool_kernel=3):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            conv_in = in_channels + (in_channels // 2)
            self.conv = SmoothedResBlock(conv_in, out_channels, smooth_skip=smooth_skip, pool_kernel=pool_kernel)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = SmoothedResBlock(in_channels, out_channels, smooth_skip=smooth_skip, pool_kernel=pool_kernel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)









# --- 1. THE BUILDING BLOCK (ResBlock + GroupNorm) ---
class ResBlock(nn.Module):
    """
    Residual Block with Group Normalization.
    Structure: Input + Conv1 -> GN -> ReLU -> Conv2 -> GN -> Output
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # We use 8 groups for GroupNorm (safe default)
        groups = 8
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        )
        
        # Skip Connection Adjustment (if channel sizes differ)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=groups, num_channels=out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.shortcut(x) + self.conv_block(x))



class AUSequentialUNet(nn.Module):
    HINT_MODES = ('subtract', 'concat', 'encoder-only')

    def __init__(self, n_channels=1, bilinear=False, gamma=1, kernels=[1, 3, 5],base_latent_dim=32, use_derivatives=True,
                 hint_mode='subtract'):
        super(AUSequentialUNet, self).__init__()

        # ABLATION: gate input mode. 'subtract' reproduces Pmodel.py exactly.
        if hint_mode not in self.HINT_MODES:
            raise ValueError(f"hint_mode must be one of {self.HINT_MODES}, got {hint_mode!r}")
        self.hint_mode = hint_mode
        # 'concat' stacks E_l and B_l, so the gate's hint input is twice as wide.
        hf = 2 if hint_mode == 'concat' else 1

        self.use_derivatives = use_derivatives
        self.gamma = gamma
        self.bilinear = bilinear
        
        # --- ENCODER ---
        augmented_channels = n_channels * 3 if self.use_derivatives else n_channels
        
        self.inc = RK4SmoothedBlock(augmented_channels, 64, smooth_skip=False, pool_kernel=3)
        self.down1 = Down(64, 128, smooth_skip=False, pool_kernel=3)
        self.down2 = Down(128, 256, smooth_skip=False, pool_kernel=3)
        self.down3 = Down(256, 512, smooth_skip=False, pool_kernel=3)
        
        # ==========================================
        # LATENT SPACE DISENTANGLEMENT (The Split)
        # ==========================================
        self.base_latent_dim = base_latent_dim
        self.sig_latent_dim = 512 - self.base_latent_dim # 480
        
        # --- CUSTOM STAGE 1 (Deep) FOR BASELINE ---
        if self.bilinear:
            self.up1_base_trans = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up1_base_trans = nn.ConvTranspose1d(self.base_latent_dim, self.base_latent_dim, kernel_size=2, stride=2)
            
        # Concat: 32 (from bottleneck) + 256 (from skip x3) = 288 in_channels
        self.up1_base_conv = SmoothedResBlock(self.base_latent_dim + 256, 256, smooth_skip=False)

        # --- CUSTOM STAGE 1 (Deep) FOR SIGNAL ---
        if self.bilinear:
            self.up1_sig_trans = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up1_sig_trans = nn.ConvTranspose1d(self.sig_latent_dim, self.sig_latent_dim, kernel_size=2, stride=2)
            
        # Concat: 480 (from bottleneck) + 256 (from skip x3) = 736 in_channels
        self.up1_sig_conv = SmoothedResBlock(self.sig_latent_dim + 256, 256, smooth_skip=False)

        # --- BASELINE HEAD (Stages 2 & 3 remain symmetric) ---
        self.up2_base = Up(256, 128, bilinear, smooth_skip=False)
        self.up3_base = Up(128, 64, bilinear, smooth_skip=False,pool_kernel=3)
        self.out_base = nn.Conv1d(64, n_channels, kernel_size=1)

        # --- BASELINE AUXILIARY HEADS (Deep Supervision) ---
        self.out_base_deep = nn.Conv1d(256, n_channels, kernel_size=1)
        self.out_base_mid = nn.Conv1d(128, n_channels, kernel_size=1)
        
        # --- SIGNAL HEAD ---
        # HETEROGENEOUS STAGE 1: Gate now expects the 480-channel signal split!
        #self.attn1 = TopDownPhysicsGate(current_channels=256, deep_channels=self.sig_latent_dim, kernel_size=kernels[0])
        gate_bias=None
        temp=1.0
        self.attn1 = HighResPhysicsGate(
                    current_channels=256, 
                    deep_channels=self.sig_latent_dim, 
                    hint_channels=256*hf,  # ABLATION: doubled in 'concat' mode
                    fusion_mode='concat',  # <--- Toggle between 'concat' and 'sum'
                    temperature=temp,
                    bias_value=gate_bias,
                    kernel_size=kernels[1]
                )
        # Stage 2: Mid (256 -> 128)
        self.up2_sig = Up(256, 128, bilinear)
        #self.attn2 = DeepHybridPhysicsGate(current_channels=128, deep_channels=256, deep_hint_channels=256, kernel_size=kernels[1],use_deep_hint=False)
        self.attn2 = HighResPhysicsGate(
                    current_channels=128, 
                    deep_channels=256, 
                    hint_channels=128*hf,  # ABLATION: doubled in 'concat' mode
                    fusion_mode='concat',  # <--- Toggle between 'concat' and 'sum'
                    temperature=temp,
                    bias_value=gate_bias,
                    kernel_size=kernels[1]
                )
        # Stage 3: Shallow (128 -> 64)
        self.up3_sig = Up(128, 64, bilinear,smooth_skip=False,pool_kernel=5)
        #self.attn3 = DeepHybridPhysicsGate(current_channels=64, deep_channels=128, deep_hint_channels=128, kernel_size=kernels[2],use_deep_hint=False)
        
        self.attn3 = HighResPhysicsGate(
                    current_channels=64, 
                    deep_channels=128, 
                    hint_channels=64*hf,  # ABLATION: doubled in 'concat' mode
                    fusion_mode='concat',  # <--- Toggle between 'concat' and 'sum'
                    temperature=temp,
                    bias_value=gate_bias,
                    kernel_size=kernels[1]
                )

        self.out_sig = nn.Conv1d(64, n_channels, kernel_size=1)

        # --- SIGNAL AUXILIARY HEADS (Deep Supervision) ---
        self.out_sig_deep = nn.Conv1d(256, n_channels, kernel_size=1)
        self.out_sig_mid = nn.Conv1d(128, n_channels, kernel_size=1)

        # Squelch Gate
        #self.gate = SquelchGate(in_channels=64, init_val=0.0)
        self.gate = SpatialSquelchGate(kernel_size=3,
                                       temperature=1.0,
                                       bias_value=None)
    
    
    def forward(self, x):
        # Augment Input on the Fly
        if self.use_derivatives:
            x_input = self.compute_derivatives(x)  
        else:
            x_input = x  

        # 1. Encoder 
        x1 = self.inc(x_input)    # 64 ch
        x2 = self.down1(x1)       # 128 ch
        x3 = self.down2(x2)       # 256 ch
        x4 = self.down3(x3)       # 512 ch

        # ==========================================
        # LATENT SPACE DISENTANGLEMENT (The Slicing)
        # ==========================================
        x4_base = x4[:, :self.base_latent_dim, :] # First 32 channels
        x4_sig = x4[:, self.base_latent_dim:, :]  # Remaining 480 channels

        # 2. Baseline Head (Stage 1 Custom)
        b_deep_up = self.up1_base_trans(x4_base)
        
        diffY_b = x3.size()[2] - b_deep_up.size()[2]
        b_deep_up = F.pad(b_deep_up, [diffY_b // 2, diffY_b - diffY_b // 2])
        
        b_deep_concat = torch.cat([x3, b_deep_up], dim=1)
        b_deep = self.up1_base_conv(b_deep_concat)     # Outputs 256 ch
        
        # Baseline Head (Stages 2 & 3)
        b_mid  = self.up2_base(b_deep, x2) # 128 ch
        b_final = self.up3_base(b_mid, x1) # 64 ch
        
        baseline = self.out_base(b_final)

        # 3. Signal Head (Stage 1 Custom)
        s_deep_up = self.up1_sig_trans(x4_sig)
        
        diffY_s = x3.size()[2] - s_deep_up.size()[2]
        s_deep_up = F.pad(s_deep_up, [diffY_s // 2, diffY_s - diffY_s // 2])
        
        s_deep_concat = torch.cat([x3, s_deep_up], dim=1)
        s_deep_raw = self.up1_sig_conv(s_deep_concat)  # Outputs 256 ch
        
        # --- STAGE 1 GATING ---
        hint_deep = self._hint(x3, b_deep)   # ABLATION
        # TopDownGate input: current_signal, deep_signal (now x4_sig), current_hint
        #s_deep = self.attn1(s_deep_raw, x4_sig, hint_deep)
        s_deep = self.attn1(current_signal=s_deep_raw, 
            deep_signal=x4_sig, 
            current_hint=hint_deep)
        # --- STAGE 2 (Mid) ---
        hint_mid = self._hint(x2, b_mid)     # ABLATION
        s_mid_up = self.up2_sig(s_deep, x2)
        #s_mid = self.attn2(s_mid_up, s_deep, hint_deep, hint_mid)

        s_mid = self.attn2(
            current_signal=s_mid_up, 
            deep_signal=s_deep, 
            current_hint=hint_mid
        )

        # --- STAGE 3 (Shallow Final) ---
        hint_final = self._hint(x1, b_final) # ABLATION
        s_final_up = self.up3_sig(s_mid, x1)
        #s_final = self.attn3(s_final_up, s_mid, hint_mid, hint_final) 

        s_final = self.attn3(current_signal=s_final_up, 
            deep_signal=s_mid, 
            current_hint=hint_final) 
        
        # Final Projection 
        raw_signal = self.out_sig(s_final)
        
        # Squelch Gate 
        final_signal = self.gate(s_final, raw_signal)

        # --- DEEP SUPERVISION: Generate Intermediate Predictions ---
        pred_s_deep = self.out_sig_deep(s_deep)
        pred_s_mid = self.out_sig_mid(s_mid)
        
        pred_b_deep = self.out_base_deep(b_deep)
        pred_b_mid = self.out_base_mid(b_mid)
        # Safely upsample all predictions to match original sequence length
        

        # RETURN 8 ITEMS (Including the slices for the orthogonality loss)
        return final_signal, baseline, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig

    def _hint(self, E, B):
        """ABLATION: what the gate receives as its 'hint'."""
        if self.hint_mode == 'subtract':
            return E - B
        if self.hint_mode == 'concat':
            return torch.cat([E, B], dim=1)
        return E  # 'encoder-only': baseline branch does not inform the gate

    def compute_derivatives(self, x):
        """
        Computes 1st and 2nd derivatives via finite differences.
        Maintains the same spatial resolution via padding.
        """
        dx = x[:, :, 1:] - x[:, :, :-1]
        dx = F.pad(dx, (1, 0)) 
        
        d2x = dx[:, :, 1:] - dx[:, :, :-1]
        d2x = F.pad(d2x, (1, 0))
        
        return torch.cat([x, dx, d2x], dim=1)





class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state automatically.
    """
    def __init__(self, patience=50, start_epoch=50, verbose=False, delta=1e-3, path='best_raman_model.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            start_epoch (int): Epoch to start monitoring (no stopping/saving before this).
            verbose (bool): If True, prints a message for each validation loss improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): Trace print function.
        """
        self.patience = patience
        self.start_epoch = start_epoch # <--- NEW PARAMETER
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta             # <--- UPDATED DEFAULT (1e-3)
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, epoch): # <--- ADDED 'epoch' ARGUMENT
        # 1. Check if we are in the "Warmup Phase"
        if epoch < self.start_epoch:
            return # Do nothing, just keep training

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            
        # 2. Check if improvement is LESS than delta (i.e., not good enough)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        # 3. Significant improvement found
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss









class ContrastiveLoss(nn.Module):
    def __init__(self, w_signal=100.0, w_base=1.0, w_consist=10.0, 
                 w_smooth=0.1, w_curve=0.1, dip_factor=0.5,
                 w_mae=1.0, w_cos=1.0, w_shape=0.2, # <--- NEW: Explicit Signal Weights
                 w_mid=50.0, w_deep=25.0, w_ortho=0.5):
        """
        Args:
            w_mae: Weight for quantitative amplitude calibration.
            w_cos: Weight for qualitative macro-geometry.
            w_shape: Weight for micro-dynamic derivative matching.
            ...
        """
        super(ContrastiveLoss, self).__init__()
        
        self.w_s = w_signal
        self.w_b = w_base
        
        self.w_c = w_consist
        self.w_tv1 = w_smooth 
        self.w_tv2 = w_curve 
        self.dip = dip_factor
        
        # Explicit Signal Physics Weights
        self.w_mae = w_mae
        self.w_cos = w_cos
        self.w_shape = w_shape 
        
        # Auxiliary & Bottleneck weights
        self.w_mid = w_mid
        self.w_deep = w_deep
        self.w_ortho = w_ortho
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    # --- Baseline Regularizers (Minimize Magnitude) ---
    def tv1_loss(self, x):
        """Baseline wants to be FLAT (Minimize slope)"""
        return torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))

    def tv2_loss(self, x):
        """Baseline wants to be SMOOTH (Minimize curvature)"""
        diff2 = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
        return torch.mean(torch.abs(diff2))

    # --- NEW HELPER: Orthogonality Penalty (Feature Decoupling) ---
    def orthogonality_loss(self, base_latent, sig_latent):
        base_norm = F.normalize(base_latent, p=2, dim=2)
        sig_norm = F.normalize(sig_latent, p=2, dim=2)
        correlation_matrix = torch.bmm(base_norm, sig_norm.transpose(1, 2))
        return torch.mean(correlation_matrix ** 2)

    # --- HELPER: Isolated Signal Loss (With Masking & Normalization) ---
    # Add the 'apply_shape' toggle
    def compute_signal_loss(self, pred_s, target_s, apply_shape=True, aux_w_mae=None, aux_w_cos=None):
        """Calculates intensity and shape loss universally across all spectra."""
        
        # Use overrides if provided (for aux layers), otherwise use the final layer weights
        current_w_mae = aux_w_mae if aux_w_mae is not None else self.w_mae
        current_w_cos = aux_w_cos if aux_w_cos is not None else self.w_cos

        # --- A. AMPLITUDE LOSS (MAE) ---
        mae_loss = self.l1(pred_s, target_s)

        # --- B. SHAPE LOSS (Cosine) ---
        pred_flat = pred_s.flatten(1)
        target_flat = target_s.flatten(1)
        cos_sims = F.cosine_similarity(pred_flat, target_flat, dim=1)
        cos_loss = (1.0 - cos_sims).mean()

        # --- C. OPTIONAL DERIVATIVE PHYSICS LOSS (Shape) ---
        shape_loss = torch.tensor(0.0, device=pred_s.device) 
        
        if apply_shape:
            d_pred_1 = pred_s[:, :, 1:] - pred_s[:, :, :-1]
            d_targ_1 = target_s[:, :, 1:] - target_s[:, :, :-1]
            loss_d1 = F.l1_loss(d_pred_1, d_targ_1)
            
            d_pred_2 = d_pred_1[:, :, 1:] - d_pred_1[:, :, :-1]
            d_targ_2 = d_targ_1[:, :, 1:] - d_targ_1[:, :, :-1]
            loss_d2 = F.l1_loss(d_pred_2, d_targ_2)
            
            shape_loss = loss_d1 + loss_d2
            
            total_weight = current_w_mae + current_w_cos + self.w_shape
            raw_signal_total = (current_w_mae * mae_loss) + (current_w_cos * cos_loss) + (self.w_shape * shape_loss)
            
        else:
            total_weight = current_w_mae + current_w_cos
            raw_signal_total = (current_w_mae * mae_loss) + (current_w_cos * cos_loss)

        normalized_signal_loss = raw_signal_total / total_weight
        
        return normalized_signal_loss, mae_loss, cos_loss, shape_loss
    
    
    # --- HELPER: Isolated Baseline Fit Loss ---
    def compute_base_loss(self, pred_b, target_b_dipped):
        """Calculates pure MSE fit for ANY baseline prediction"""
        return self.mse(pred_b, target_b_dipped)
    
    def forward(self, pred_s, target_s, pred_b, target_b, input_noisy, 
                pred_s_mid=None, pred_s_deep=None, 
                pred_b_mid=None, pred_b_deep=None,
                x4_base=None, x4_sig=None): 
        
        # --- 1. SIGNAL LOSS (Final) ---
        loss_s, mae_loss, cos_loss, shape_loss = self.compute_signal_loss(pred_s, target_s)

        # --- 2. BASELINE LOSS (Final) ---
        physical_target_s = target_s / 10.0
        target_b_dipped = target_b - (self.dip * physical_target_s)
        
        fit_loss = self.compute_base_loss(pred_b, target_b_dipped)
        loss_tv1 = self.tv1_loss(pred_b)
        loss_tv2 = self.tv2_loss(pred_b)
        
        loss_b = fit_loss + (self.w_tv1 * loss_tv1) + (self.w_tv2 * loss_tv2)

        # --- 3. DEEP SUPERVISION LOSS ---
        loss_aux_s = 0.0
        loss_aux_b = 0.0

        # Define the specific auxiliary loss ratios (Heavy MAE focus)
        AUX_COS = 0.2
        AUX_MAE = 2.0-AUX_COS
        
        
        if pred_s_mid is not None and pred_b_mid is not None:
            # Calculate how much to pool (e.g., 864 / 216 = 4)
            pool_factor_mid = target_s.shape[-1] // pred_s_mid.shape[-1]
            
            # Downsample the targets using Average Pooling
            target_s_mid = F.avg_pool1d(target_s, kernel_size=pool_factor_mid, stride=pool_factor_mid)
            target_b_mid = F.avg_pool1d(target_b_dipped, kernel_size=pool_factor_mid, stride=pool_factor_mid)
            
            # Compare native vs native
            l_s_mid, _, _, _ = self.compute_signal_loss(pred_s_mid, target_s_mid, apply_shape=False, aux_w_mae=AUX_MAE, aux_w_cos=AUX_COS)
            l_b_mid = self.compute_base_loss(pred_b_mid, target_b_mid)
            
            loss_aux_s += self.w_mid * l_s_mid
            loss_aux_b += self.w_mid * l_b_mid 
            
        if pred_s_deep is not None and pred_b_deep is not None:
            # Calculate how much to pool (e.g., 864 / 108 = 8)
            pool_factor_deep = target_s.shape[-1] // pred_s_deep.shape[-1]
            
            # Downsample the targets using Average Pooling
            target_s_deep = F.avg_pool1d(target_s, kernel_size=pool_factor_deep, stride=pool_factor_deep)
            target_b_deep = F.avg_pool1d(target_b_dipped, kernel_size=pool_factor_deep, stride=pool_factor_deep)
            
            # Compare native vs native
            l_s_deep, _, _, _ = self.compute_signal_loss(pred_s_deep, target_s_deep, apply_shape=False, aux_w_mae=AUX_MAE, aux_w_cos=AUX_COS)
            l_b_deep = self.compute_base_loss(pred_b_deep, target_b_deep)
            
            loss_aux_s += self.w_deep * l_s_deep
            loss_aux_b += self.w_deep * l_b_deep

        # --- 4. CONSISTENCY ---
        physical_pred_s = pred_s / 10.0
        recon = pred_b + (physical_pred_s * (1.0 + self.dip))
        loss_c = self.mse(recon, input_noisy)

        # --- 5. LATENT SPACE DISENTANGLEMENT (Orthogonality) ---
        loss_ortho = 0.0
        if x4_base is not None and x4_sig is not None:
            loss_ortho = self.orthogonality_loss(x4_base, x4_sig)

        # --- 6. MACRO TASKS ---
        raw_signal_loss = loss_s + loss_aux_s
        raw_base_loss = loss_b + loss_aux_b + (self.w_c * loss_c) + (self.w_ortho * loss_ortho)

        
        total_loss = (self.w_s * raw_signal_loss) + (self.w_b * raw_base_loss)
        
        # --- 7. SAFELY EXTRACT AUX/ORTHO VALUES ---
        # We need to make sure we extract the scalar float value whether it's a tensor or a 0.0 float
        aux_s_val = loss_aux_s.item() if isinstance(loss_aux_s, torch.Tensor) else loss_aux_s
        aux_b_val = loss_aux_b.item() if isinstance(loss_aux_b, torch.Tensor) else loss_aux_b
        ortho_val = loss_ortho.item() if isinstance(loss_ortho, torch.Tensor) else loss_ortho

        # Returning 10 items now to feed the experiment logger and dynamics plot!
        return total_loss, (mae_loss.item(), cos_loss.item(), shape_loss.item(), 
                            fit_loss.item(), loss_tv1.item(), loss_tv2.item(), loss_c.item(),
                            aux_s_val, aux_b_val, ortho_val)





class ContrastiveLoss_nomask(nn.Module):
    def __init__(self, w_signal=100.0, w_base=1.0, w_consist=10.0, 
                 w_smooth=0.1, w_curve=0.1, dip_factor=0.5,
                 w_mae=1.0, w_cos=1.0, w_shape=0.2, # <--- RESTORED: w_cos
                 w_mid=50.0, w_deep=25.0, w_ortho=0.5):
        """
        Args:
            w_mae: Weight for quantitative amplitude calibration.
            w_cos: Weight for qualitative macro-geometry.
            w_shape: Weight for micro-dynamic derivative matching.
            ...
        """
        super(ContrastiveLoss_nomask, self).__init__()
        
        self.w_s = w_signal
        self.w_b = w_base
        
        self.w_c = w_consist
        self.w_tv1 = w_smooth 
        self.w_tv2 = w_curve 
        self.dip = dip_factor
        
        # Explicit Signal Physics Weights
        self.w_mae = w_mae
        self.w_cos = w_cos
        self.w_shape = w_shape 
        
        # Auxiliary & Bottleneck weights
        self.w_mid = w_mid
        self.w_deep = w_deep
        self.w_ortho = w_ortho
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    # --- Baseline Regularizers (Minimize Magnitude) ---
    def tv1_loss(self, x):
        """Baseline wants to be FLAT (Minimize slope)"""
        return torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))

    def tv2_loss(self, x):
        """Baseline wants to be SMOOTH (Minimize curvature)"""
        diff2 = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
        return torch.mean(torch.abs(diff2))

    # --- HELPER: Orthogonality Penalty (Feature Decoupling) ---
    def orthogonality_loss(self, base_latent, sig_latent):
        base_norm = F.normalize(base_latent, p=2, dim=2)
        sig_norm = F.normalize(sig_latent, p=2, dim=2)
        correlation_matrix = torch.bmm(base_norm, sig_norm.transpose(1, 2))
        return torch.mean(correlation_matrix ** 2)

    # --- HELPER: Isolated Signal Loss (UNMASKED) ---
    def compute_signal_loss(self, pred_s, target_s):
        """Calculates intensity and shape loss universally across all spectra."""
        
        # --- A. AMPLITUDE LOSS (MAE) ---
        mae_loss = self.l1(pred_s, target_s)

        # --- B. SHAPE LOSS (Cosine) ---
        pred_flat = pred_s.flatten(1)
        target_flat = target_s.flatten(1)
        cos_sims = F.cosine_similarity(pred_flat, target_flat, dim=1)
        # Unmasked: Just take the standard mean of all cosine errors
        cos_loss = (1.0 - cos_sims).mean()

        # --- C. DERIVATIVE PHYSICS LOSS (Shape) ---
        d_pred_1 = pred_s[:, :, 1:] - pred_s[:, :, :-1]
        d_targ_1 = target_s[:, :, 1:] - target_s[:, :, :-1]
        # Unmasked: PyTorch l1_loss automatically uses 'mean' reduction
        loss_d1 = F.l1_loss(d_pred_1, d_targ_1)
        
        d_pred_2 = d_pred_1[:, :, 1:] - d_pred_1[:, :, :-1]
        d_targ_2 = d_targ_1[:, :, 1:] - d_targ_1[:, :, :-1]
        loss_d2 = F.l1_loss(d_pred_2, d_targ_2)
        
        shape_loss = loss_d1 + loss_d2

        # --- D. WEIGHTED NORMALIZATION ---
        total_weight = self.w_mae + self.w_cos + self.w_shape
        
        raw_signal_total = (self.w_mae * mae_loss) + (self.w_cos * cos_loss) + (self.w_shape * shape_loss)
        normalized_signal_loss = raw_signal_total / total_weight
        
        return normalized_signal_loss, mae_loss, cos_loss, shape_loss
    
    # --- HELPER: Isolated Baseline Fit Loss ---
    def compute_base_loss(self, pred_b, target_b_dipped):
        """Calculates pure MSE fit for ANY baseline prediction"""
        return self.mse(pred_b, target_b_dipped)
    
    def forward(self, pred_s, target_s, pred_b, target_b, input_noisy, 
                pred_s_mid=None, pred_s_deep=None, 
                pred_b_mid=None, pred_b_deep=None,
                x4_base=None, x4_sig=None): 
        
        # --- 1. SIGNAL LOSS (Final) ---
        loss_s, mae_loss, cos_loss, shape_loss = self.compute_signal_loss(pred_s, target_s)

        # --- 2. BASELINE LOSS (Final) ---
        physical_target_s = target_s / 10.0
        target_b_dipped = target_b - (self.dip * physical_target_s)
        
        fit_loss = self.compute_base_loss(pred_b, target_b_dipped)
        loss_tv1 = self.tv1_loss(pred_b)
        loss_tv2 = self.tv2_loss(pred_b)
        
        loss_b = fit_loss + (self.w_tv1 * loss_tv1) + (self.w_tv2 * loss_tv2)

        # --- 3. DEEP SUPERVISION LOSS ---
        loss_aux_s = 0.0
        loss_aux_b = 0.0
        
        if pred_s_mid is not None and pred_b_mid is not None:
            l_s_mid, _, _, _ = self.compute_signal_loss(pred_s_mid, target_s)
            l_b_mid = self.compute_base_loss(pred_b_mid, target_b_dipped)
            loss_aux_s += self.w_mid * l_s_mid
            loss_aux_b += self.w_mid * l_b_mid 
            
        if pred_s_deep is not None and pred_b_deep is not None:
            l_s_deep, _, _, _ = self.compute_signal_loss(pred_s_deep, target_s)
            l_b_deep = self.compute_base_loss(pred_b_deep, target_b_dipped)
            loss_aux_s += self.w_deep * l_s_deep
            loss_aux_b += self.w_deep * l_b_deep

        # --- 4. CONSISTENCY ---
        physical_pred_s = pred_s / 10.0
        recon = pred_b + (physical_pred_s * (1.0 + self.dip))
        loss_c = self.mse(recon, input_noisy)

        # --- 5. LATENT SPACE DISENTANGLEMENT (Orthogonality) ---
        loss_ortho = 0.0
        if x4_base is not None and x4_sig is not None:
            loss_ortho = self.orthogonality_loss(x4_base, x4_sig)

        # --- 6. MACRO TASKS ---
        raw_signal_loss = loss_s + loss_aux_s
        raw_base_loss = loss_b + loss_aux_b + (self.w_c * loss_c) + (self.w_ortho * loss_ortho)
        
        total_loss = (self.w_s * raw_signal_loss) + (self.w_b * raw_base_loss)
        
        # --- 7. SAFELY EXTRACT AUX/ORTHO VALUES ---
        aux_s_val = loss_aux_s.item() if isinstance(loss_aux_s, torch.Tensor) else loss_aux_s
        aux_b_val = loss_aux_b.item() if isinstance(loss_aux_b, torch.Tensor) else loss_aux_b
        ortho_val = loss_ortho.item() if isinstance(loss_ortho, torch.Tensor) else loss_ortho

        # Returning exactly 10 items again so your train_model unpacking stays perfectly aligned!
        return total_loss, (mae_loss.item(), cos_loss.item(), shape_loss.item(), 
                            fit_loss.item(), loss_tv1.item(), loss_tv2.item(), loss_c.item(),
                            aux_s_val, aux_b_val, ortho_val)







def visualize_spatial_gates(model, dataloader, device, num_samples=3):
    """
    Uses PyTorch Forward Hooks to extract and visualize the 1D spatial 
    attention masks from the network's HighResPhysicsGate modules.
    """
    print("Deploying Forward Hooks to extract Attention Gates...")
    model.eval()
    
    # 1. Setup the Dictionary and Hook Function
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            # output is the tensor right after the Sigmoid (the 0.0 to 1.0 mask)
            activations[name] = output.detach().cpu().numpy()
        return hook

    # 2. Attach hooks to every Gate in the network
    hooks = []
    for name, module in model.named_modules():
        # Look for your custom gate classes
        if 'HighResPhysicsGate' in type(module).__name__ or 'SpatialSquelchGate' in type(module).__name__:
            # We specifically hook onto the 'sigmoid' layer you defined in __init__
            if hasattr(module, 'sigmoid'):
                handle = module.sigmoid.register_forward_hook(get_activation(name))
                hooks.append(handle)
                print(f"Attached probe to: {name}")

    if not hooks:
        print("Warning: No gates found. Make sure your model uses the exact class names.")
        return

    # 3. Run a single batch of data through the probed network
    with torch.no_grad():
        inputs, target_s, target_b = next(iter(dataloader))
        inputs = inputs.to(device)
        
        # The forward pass triggers our hooks automatically!
        pred_s, pred_b, *_ = model(inputs)
        
        # Move data to CPU for plotting
        inputs_np = inputs.cpu().numpy()
        pred_s_np = pred_s.cpu().numpy()
        target_s_np = target_s.cpu().numpy()

    # 4. Remove the hooks so they don't slow down future training
    for h in hooks:
        h.remove()

    # ==========================================
    # PLOTTING THE HEATMAPS
    # ==========================================
    # Find the deepest/highest-resolution gate to plot
    gate_names = list(activations.keys())
    target_gate_name = gate_names[-1] # Usually the last gate in the decoder
    gate_masks = activations[target_gate_name] # Shape: [Batch, 1, Length]

    print(f"\nVisualizing Mask from: {target_gate_name}")

    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 4 * num_samples), dpi=150)
    if num_samples == 1:
        axes = [axes] # Handle indexing for single sample

    for i in range(num_samples):
        x_axis = np.arange(inputs_np.shape[-1])
        
        raw_noisy = inputs_np[i, 0, :]
        true_sig = target_s_np[i, 0, :]
        pred_sig = pred_s_np[i, 0, :]
        mask = gate_masks[i, 0, :] # The 0.0 to 1.0 Squelch Mask

        # --- PANEL A: Raw Input vs Output ---
        ax1 = axes[i][0]
        ax1.plot(x_axis, raw_noisy, color='lightgray', label='Noisy Input')
        ax1.plot(x_axis, true_sig, color='black', linestyle='--', label='Ground Truth')
        ax1.plot(x_axis, pred_sig, color='crimson', linewidth=2, label='Predicted Signal')
        ax1.set_title(f"Sample {i+1}: Spectra Reconstruction", fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # --- PANEL B: The Attention Mask Heatmap ---
        ax2 = axes[i][1]
        
        # Plot the predicted signal again for reference
        ax2.plot(x_axis, pred_sig, color='black', linewidth=1.5, label='Extracted Signal')
        
        # Overlay the mask using a secondary Y-axis
        ax3 = ax2.twinx()
        ax3.fill_between(x_axis, 0, mask, color='gold', alpha=0.4, label='Gate Probability (Mask)')
        ax3.plot(x_axis, mask, color='darkorange', linewidth=2, linestyle=':')
        
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_ylabel("Gate Probability [0.0 - 1.0]", color='darkorange', fontweight='bold')
        
        ax2.set_title(f"Internal Physics: {target_gate_name} Mask", fontweight='bold')
        ax2.set_xlabel("Raman Shift (Pixels)")
        
        # Combine legends
        lines_1, labels_1 = ax2.get_legend_handles_labels()
        lines_2, labels_2 = ax3.get_legend_handles_labels()
        ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    plt.savefig("Spatial_Gate_Heatmaps.png", dpi=300)
    plt.show()
    print("Saved 'Spatial_Gate_Heatmaps.png'.")