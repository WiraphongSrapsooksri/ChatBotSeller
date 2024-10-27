# training_visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

class TrainingVisualizer:
    def __init__(self, history, save_dir='plots'):
        self.history = history
        self.save_dir = save_dir
        Path(save_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        
    def plot_training_metrics(self):
        """พล็อตกราฟ accuracy และ loss"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', pad=15, size=12)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss', pad=15, size=12)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_learning_rate(self):
        """พล็อตกราฟ learning rate ถ้ามี"""
        if 'lr' in self.history.history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history.history['lr'], linewidth=2)
            plt.title('Learning Rate Over Time', pad=15, size=12)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            plt.savefig(f'{self.save_dir}/learning_rate.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """พล็อต confusion matrix"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix', pad=15, size=12)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_metrics_summary(self, y_true, y_pred, labels):
        """บันทึกสรุปผลการเทรน"""
        report = classification_report(y_true, y_pred, target_names=labels)
        
        metrics_summary = {
            'final_accuracy': self.history.history['accuracy'][-1],
            'final_val_accuracy': self.history.history['val_accuracy'][-1],
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
            'best_val_accuracy': max(self.history.history['val_accuracy']),
            'best_epoch': np.argmax(self.history.history['val_accuracy']) + 1,
            'total_epochs': len(self.history.history['accuracy'])
        }
        
        with open(f'{self.save_dir}/training_summary.txt', 'w', encoding='utf-8') as f:
            f.write("=== Training Summary ===\n\n")
            f.write("Model Performance Metrics:\n")
            f.write("-" * 30 + "\n")
            for metric, value in metrics_summary.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nClassification Report:\n")
            f.write("-" * 30 + "\n")
            f.write(report)