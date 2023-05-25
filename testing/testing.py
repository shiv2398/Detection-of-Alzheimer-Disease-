
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import torch
import os 
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from files.utils import TRAIN_CONFIG
from data_preparation.class_loader import Simple_Dataset

def test(model, criterion, test_loader, device=None):
    print('Model_Testing:-')
    model.eval()
    if TRAIN_CONFIG['testing']['metrics']['loss']:
        test_loss = 0.0
    if TRAIN_CONFIG['testing']['metrics']['accuracy']:
        correct = 0
        total = 0
    if TRAIN_CONFIG['testing']['metrics']['f1_score']:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    if TRAIN_CONFIG['testing']['metrics']['save']:
        test_loss /= len(test_loader)
        accuracy = correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        specificity = tn / (tn + fp + 1e-10)
        npv = tn / (tn + fn + 1e-10)
        prevalence = (tp + fn) / (tp + tn + fp + fn + 1e-10)
        lr_plus = recall / (1 - specificity + 1e-10)
        lr_minus = (1 - recall + 1e-10) / specificity
        test_metrics = {'Test Loss': test_loss, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 
                        'F1 Score': f1, 'Specificity': specificity, 'NPV': npv, 'Prevalence': prevalence, 
                        'LR+': lr_plus, 'LR-': lr_minus}

        return test_metrics
    else:
        test_loss /= len(test_loader)
        accuracy = correct / total