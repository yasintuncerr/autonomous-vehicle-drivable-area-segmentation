from src.utils.postprocess import denormalize_image, postprocess_output, visualize_results, visualize_overlay_comparison, visualize_overlay 
from src.utils.preprocess import create_pointcloud_image, create_morphological_polygon
from src.model import UNetModel, UNetConfig, create_unet_model
from src.utils.dataset import MultiViewImageDataset
from src.utils.loss import DiceLoss, FocalLoss, CombinedLoss, IoULoss
from src.utils.metrics import SegmentationMetrics
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import os
import argparse


class Prediction:
    def __init__(self, model_path):
        print(f"Loading model from: {model_path}")
        
        # Load model
        self.model = UNetModel.from_pretrained(model_path)
        self.model.eval()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")
        
        # Get model configuration
        try:
            self.width = self.model.config.input_width
            self.height = self.model.config.input_height
        except AttributeError:
            # Default values if config doesn't have these attributes
            print("Warning: Model config doesn't have input dimensions. Using default 256x256")
            self.width = 256
            self.height = 256
        
        print(f"Input dimensions: {self.width}x{self.height}")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    def predict(self, image, pc_image=None):
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Handle point cloud image
        if pc_image is None:
            # Create empty point cloud image
            pc_image = Image.new('RGB', (self.width, self.height), color=(0, 0, 0))
        elif isinstance(pc_image, np.ndarray):
            pc_image = Image.fromarray(pc_image)
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        pc_image_tensor = self.transform(pc_image).unsqueeze(0).to(self.device)
        
        # Concatenate inputs along channel dimension
        input_tensor = torch.cat((image_tensor, pc_image_tensor), dim=1)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert to numpy
        output = output.squeeze(0).cpu().numpy()
        
        return output
    
    def predict_and_visualize(self, image, pc_image=None, opacity=0.5, color=(0, 255, 0)):
        # Make prediction
        output = self.predict(image, pc_image)
        
        # Create overlay
        overlayed_image = visualize_overlay(
            image=image, 
            mask=output, 
            opacity=opacity, 
            color=color
        )
        
        return output, overlayed_image


def load_image_safely(image_path, default_size=(398, 224)):
    if not image_path:
        return None
        
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            print(f"Successfully loaded image: {image_path}")
            return image
        else:
            print(f"Image not found: {image_path}")
            return None
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None


def save_results(output, overlayed_image, output_path, base_name="prediction"):
    os.makedirs(output_path, exist_ok=True)
    
    # Save overlay image
    overlay_path = os.path.join(output_path, f"{base_name}_overlay.png")
    if isinstance(overlayed_image, np.ndarray):
        overlayed_pil = Image.fromarray(overlayed_image)
        overlayed_pil.save(overlay_path)
    else:
        overlayed_image.save(overlay_path)
    print(f"Overlay image saved: {overlay_path}")
    
    # Save raw prediction mask
    mask_path = os.path.join(output_path, f"{base_name}_mask.png")
    
    # Process output for saving
    if len(output.shape) == 3:
        if output.shape[0] == 1:
            output = output.squeeze(0)
        elif output.shape[0] > 1:
            output = output[0]  # Take first channel
    
    # Convert to 0-255 range
    if output.max() <= 1.0:
        mask_image = (output * 255).astype(np.uint8)
    else:
        mask_image = output.astype(np.uint8)
    
    mask_pil = Image.fromarray(mask_image, mode='L')
    mask_pil.save(mask_path)
    print(f"Mask image saved: {mask_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict using a pre-trained UNet model")
    parser.add_argument("--model_path", type=str, 
                       default="Yasintuncer/autonomous-vehicle-drivable-area-segmentation",
                       help="Path to the pre-trained UNet model or HuggingFace model name")
    parser.add_argument("--image_path", type=str, 
                       help="Path to the input image for prediction")
    parser.add_argument("--pc_image_path", type=str, 
                       help="Path to the point cloud data for prediction")
    parser.add_argument("--output_path", type=str, default="./outputs", 
                       help="Path to save the output prediction images")
    parser.add_argument("--opacity", type=float, default=0.5,
                       help="Overlay opacity (0.0-1.0)")
    parser.add_argument("--color", type=str, default="0,255,0",
                       help="Overlay color as R,G,B (e.g., '255,0,0' for red)")
    parser.add_argument("--show", action="store_true",
                       help="Show the result image")
    parser.add_argument("--save", action="store_true", default=True,
                       help="Save the result images")
    
    args = parser.parse_args()

    # Parse color
    try:
        color = tuple(map(int, args.color.split(',')))
        if len(color) != 3:
            raise ValueError
    except:
        print("Invalid color format. Using default green (0,255,0)")
        color = (0, 255, 0)

    # Validate inputs
    if not args.image_path and not args.pc_image_path:
        print("Warning: No input images provided. Please provide --image_path and/or --pc_image_path")
        return

    try:
        # Initialize model
        predictor = Prediction(args.model_path)
        
        # Load images
        image = load_image_safely(args.image_path)
        pc_image = load_image_safely(args.pc_image_path)
        
        if image is None and pc_image is None:
            print("Error: No valid images could be loaded.")
            return
        
        # Use main image if available, otherwise use point cloud image
        main_image = image if image is not None else pc_image
        
        # Make prediction and create visualization
        print("Making prediction...")
        output, overlayed_image = predictor.predict_and_visualize(
            image=main_image,
            pc_image=pc_image,
            opacity=args.opacity,
            color=color
        )
        
        print(f"Prediction completed. Output shape: {output.shape}")
        
        # Save results
        if args.save:
            save_results(output, overlayed_image, args.output_path)
        
        # Show results
        if args.show:
            if isinstance(overlayed_image, np.ndarray):
                overlayed_pil = Image.fromarray(overlayed_image)
                overlayed_pil.show()
            else:
                overlayed_image.show()
        
        print("Prediction completed successfully!")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()