import torch
import open_clip
import math
'''import torch.nn as nn
import torch.nn.functional as F

class Fitting(nn.Module):
    def __init__(self):
        super().__init__()
        BIOCLIP_DIM = 768   # Input size from BioCLIP's CLS token and patch features
        ANIMER_DIM = 1280   # Target size expected by AniMer's existing heads/decoder

        self.cls_token_projection = nn.Linear(BIOCLIP_DIM, ANIMER_DIM)
        self.patch_feature_projection = nn.Conv2d(BIOCLIP_DIM, ANIMER_DIM, kernel_size=1)

    def forward(self,feats,cls):
        new_cls = self.cls_token_projection(cls)
        projected_feats_channels = self.patch_feature_projection(feats)
        
        H_target, W_target = 16, 12 
        projected_feats = F.interpolate(
            projected_feats_channels,
            size=(H_target, W_target),
            mode='bicubic', 
            align_corners=False
        )
        return projected_feats,new_cls
        '''
class BioCLIPBackbone:
    def __init__(self):
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        self.model = model

    def get_bioclip_vit_features(self,image_input,device):
        """
        Extract the final raw token sequence (Patch Embeddings and CLS token)
        from the OpenCLIP ViT before the final LayerNorm and projection.
        """
        self.model.to(device)
        bioclip_vit = self.model.visual
        #print(image_input.dtype)
        #image_input = image_input.float()
        #print(image_input.shape)
        #1. Patch Embedding (conv1) ---
        # (B, 3, H, W) -> (B, 768, H/16, W/16)
        x = bioclip_vit.conv1(image_input)

        B = x.shape[0]
        D_token, H_new, W_new = x.shape[1], x.shape[2], x.shape[3]
        N_patches_new = H_new * W_new # This will be 192 in your case (16x12)
        # Flatten spatial dims and permute to (B, N_patches, D_token)
        # N_patches = (H/16) * (W/16)
        # Shape becomes (B, 768, 196) for 224x224 input
        x = x.reshape(B,D_token, -1) 
        x = x.permute(0, 2, 1) # -> (B, N_patches, 768)

        # Apply Patch Dropout (Identity in your case, but included for completeness)
        x = bioclip_vit.patch_dropout(x)
        
        #2. Add CLS Token & Positional Embedding ---
        cls_token_param = bioclip_vit.class_embedding #(768)
        pos_embed_param = bioclip_vit.positional_embedding

        # Add two dimensions to make it (1, 1, 768)
        cls_token_reshaped = cls_token_param.reshape(1, 1, -1) 

        # Expand only the batch dimension B.
        cls_token_expanded = cls_token_reshaped.expand(B, -1, -1) 

        # Concatenate with the patch embeddings 'x'
        x = torch.cat([cls_token_expanded, x], dim=1)
        
        if pos_embed_param is not None:
            x = x + pos_embed_param # Shape: (B, N_patches + 1, 768)
            
        x = bioclip_vit.ln_pre(x)
        
        #3. Transformer Blocks (The Core) ---
        x = bioclip_vit.transformer(x)
        
        #4. Final Extraction & Reshaping (AniMer Format) ---
                
        # CLS Token (first element)
        cls = x[:, 0]  # Shape: (B, 768)
        
        # Patch Tokens (rest of the sequence)
        patch_tokens = x[:, 1:] # Shape: (B, N_patches, 768)
        
        # Reshape Patch Tokens to (B, C, H, W) format 
        B, N_patches, D_token = patch_tokens.shape
        Hp = Wp = int(math.sqrt(N_patches)) #square image, e.g., 14x14 = 196 patches
        
        # Permute (B, N, C) -> (B, C, N) and reshape
        xp = patch_tokens.permute(0, 2, 1).reshape(B, D_token, Hp, Wp).contiguous()
        
        return xp, cls
