import numpy as np
from PIL import Image, ImageDraw


def preprocess_coloring(coloring):
    coloring = np.array(coloring)

    if coloring.ndim == 1:
        # Eğer tek kanal ise → gri tona çevir
        if coloring.shape[0] == 1:
            gray = int(coloring[0] * 255)
            return (gray, gray, gray)
        elif coloring.shape[0] == 3 or coloring.shape[0] == 4:
            return tuple((coloring[:3] * 255).astype(int))
        else:
            raise ValueError(f"Unsupported 1D coloring shape: {coloring.shape}")
    elif coloring.ndim == 2:
        return preprocess_coloring(coloring[0])  # ilk satırı işle
    else:
        raise ValueError("Coloring array must be 1D or 2D.")


def create_pointcloud_image(points, coloring, image_size, dot_size=1, ignore_white=True):
    # Create a new image
    result_image = Image.new('RGB', image_size, (0, 0, 0))
    draw = ImageDraw.Draw(result_image)
    
    # Extract x, y coordinates
    x_coords = points[0, :]
    y_coords = points[1, :]
    
    # Draw each point
    for i in range(len(x_coords)):
        x, y = int(x_coords[i]), int(y_coords[i])
        
        # Check if point is within image bounds
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            # Convert color to RGB tuple (assuming coloring is in 0-1 range)
            try:
                color = preprocess_coloring(coloring[i])
            except ValueError as e:
                print(f"Error processing color for point {i}: {e}")
                continue
            # Ignore white points
            if ignore_white:
                if color == (255, 255, 255):
                    continue
            # Draw circle/ellipse for the point
            if dot_size <= 1:
                draw.point((x, y), fill=color)
            else:
                radius = dot_size // 2
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
    
    return result_image


def create_morphological_polygon(points, coloring, image_size, radius=10, ignore_white=True):
    """
    Morphological operations kullanarak - çok etkili
    """
    # Filter valid points
    valid_points = []
    for i in range(len(points[0])):
        x, y = int(points[0, i]), int(points[1, i])
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            try:
                color = preprocess_coloring(coloring[i])
            except ValueError as e:
                print(f"Error processing color for point {i}: {e}")
                continue
            if ignore_white and color == (255, 255, 255):
                continue

            valid_points.append([x, y])
    
    if len(valid_points) < 3:
        return Image.new('L', image_size, 0)
    
    # Create initial point mask
    point_mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(point_mask)
    
    for point in valid_points:
        x, y = int(point[0]), int(point[1])
        draw.ellipse([x-2, y-2, x+2, y+2], fill=255)
    
    # Convert to numpy for morphological operations
    mask_array = np.array(point_mask)
    
    # Simple dilation
    from scipy.ndimage import binary_dilation, binary_fill_holes
    
    # Create circular structuring element
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    se = x*x + y*y <= radius*radius
    
    # Dilate to connect nearby points
    dilated = binary_dilation(mask_array, structure=se)
    
    # Fill holes inside the dilated shape - KEY ENHANCEMENT!
    filled = binary_fill_holes(dilated)
    
    # Convert back to uint8
    result_mask = filled.astype(np.uint8) * 255
    
    return Image.fromarray(result_mask)