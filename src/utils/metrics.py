import torch
import torch.nn.functional as F

class SegmentationMetrics:
    def __init__(self, threshold=0.5, smooth=1e-8):
        self.threshold = threshold
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        self.total_iou = 0
        self.total_dice = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_f1 = 0
        self.total_pixel_acc = 0
        self.count = 0
    
    def update(self, pred, target):
        """
        pred: model output (logits) [B, 1, H, W]
        target: ground truth mask [B, 1, H, W] 
        """
        pred = torch.sigmoid(pred)
        pred_binary = (pred > self.threshold).float()
        
        batch_size = pred.size(0)
        
        for i in range(batch_size):
            pred_flat = pred_binary[i].view(-1)
            target_flat = target[i].view(-1)
            
            # IoU
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum() - intersection
            iou = (intersection + self.smooth) / (union + self.smooth)
            
            # Dice
            dice = (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
            
            # Precision, Recall, F1
            tp = intersection
            fp = pred_flat.sum() - intersection
            fn = target_flat.sum() - intersection
            
            precision = (tp + self.smooth) / (tp + fp + self.smooth)
            recall = (tp + self.smooth) / (tp + fn + self.smooth)
            f1 = 2 * (precision * recall) / (precision + recall + self.smooth)
            
            # Pixel Accuracy
            correct_pixels = (pred_flat == target_flat).sum()
            total_pixels = pred_flat.numel()
            pixel_acc = correct_pixels / total_pixels
            
            # Update totals
            self.total_iou += iou
            self.total_dice += dice
            self.total_precision += precision
            self.total_recall += recall
            self.total_f1 += f1
            self.total_pixel_acc += pixel_acc
            self.count += 1
    
    def compute(self):
        if self.count == 0:
            return {}
        
        return {
            'IoU': (self.total_iou / self.count).item(),
            'Dice': (self.total_dice / self.count).item(),
            'Precision': (self.total_precision / self.count).item(),
            'Recall': (self.total_recall / self.count).item(),
            'F1': (self.total_f1 / self.count).item(),
            'PixelAcc': (self.total_pixel_acc / self.count).item()
        }