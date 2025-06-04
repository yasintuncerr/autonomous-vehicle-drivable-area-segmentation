import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, Union


class UNetConfig(PretrainedConfig):
    """UNet model configuration"""
    
    model_type = "unet"
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: list = [64, 128, 256, 512],
        input_height: int = 398,
        input_width: int = 224,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.input_height = input_height
        self.input_width = input_width


class DoubleConv(nn.Module):
    """Double convolution block used in UNet"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetModel(PreTrainedModel):
    """UNet model that inherits from PreTrainedModel for HuggingFace compatibility"""
    
    config_class = UNetConfig
    
    def __init__(self, config: UNetConfig):
        super().__init__(config)
        
        self.config = config
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNet
        in_channels = config.in_channels
        for feature in config.features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Up part of UNet
        for feature in reversed(config.features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Bottleneck
        self.bottleneck = DoubleConv(config.features[-1], config.features[-1]*2)
        
        # Final convolution
        self.final_conv = nn.Conv2d(config.features[0], config.out_channels, kernel_size=1)
    
    def forward(self, pixel_values, **kwargs):
        """
        Forward pass of the UNet model
        
        Args:
            pixel_values: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        skip_connections = []
        
        # Encoder (Down sampling)
        for down in self.downs:
            pixel_values = down(pixel_values)
            skip_connections.append(pixel_values)
            pixel_values = self.pool(pixel_values)
        
        # Bottleneck
        pixel_values = self.bottleneck(pixel_values)
        skip_connections = skip_connections[::-1]
        
        # Decoder (Up sampling)
        for idx in range(0, len(self.ups), 2):
            pixel_values = self.ups[idx](pixel_values)
            skip_connection = skip_connections[idx//2]
            
            # Handle dimension mismatch
            if pixel_values.shape != skip_connection.shape:
                pixel_values = F.interpolate(
                    pixel_values, 
                    size=skip_connection.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            concat_skip = torch.cat((skip_connection, pixel_values), dim=1)
            pixel_values = self.ups[idx+1](concat_skip)
        
        return self.final_conv(pixel_values)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


# Model factory function
def create_unet_model(
    in_channels: int = 3,
    out_channels: int = 1,
    features: list = [64, 128, 256, 512],
    input_height: int = 398,
    input_width: int = 224
) -> UNetModel:
    """Create a UNet model with specified configuration"""
    config = UNetConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        input_height=input_height,
        input_width=input_width
    )
    return UNetModel(config)


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_unet_model()
    
    # Test with random input (398x224x3)
    x = torch.randn(1, 3, 398, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Save model
    model.save_pretrained("./unet_model")
    print("Model saved successfully!")
    
    # Load model
    loaded_model = UNetModel.from_pretrained("./unet_model")
    print("Model loaded successfully!")