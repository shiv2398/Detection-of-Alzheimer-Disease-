import torch 
from tqdm import tqdm
from files.utils import TRAIN_CONFIG
def train(model, optimizer, criterion, train_loader, device=None):
    print('Model_Training:-')
    model.train()
    if TRAIN_CONFIG['training']['metrics']['loss']:
        train_loss = 0.0
    if TRAIN_CONFIG['training']['metrics']['accuracy']:
        correct = 0
        total = 0
    if TRAIN_CONFIG['training']['metrics']['f1_score']:
        tp = 0
        tn = 0
        fp = 0
        fn = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        tp += ((predicted == 1) & (labels == 1)).sum().item()
        tn += ((predicted == 0) & (labels == 0)).sum().item()
        fp += ((predicted == 1) & (labels == 0)).sum().item()
        fn += ((predicted == 0) & (labels == 1)).sum().item()
    if TRAIN_CONFIG['training']['metrics']['save']:
        train_loss /= len(train_loader)
        accuracy = correct / total
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        npv = tn / (tn + fn + 1e-10)
        prevalence = (tp + fn) / (tp + tn + fp + fn + 1e-10)
        lr_plus = recall / (1 - specificity + 1e-10)
        lr_minus = (1 - recall + 1e-10) / specificity
        train_metrics = {'Train Loss': train_loss, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 
                        'F1 Score': f1, 'Specificity': specificity, 'NPV': npv, 'Prevalence': prevalence, 
                        'LR+': lr_plus, 'LR-': lr_minus}

        return train_metrics
    else:
        train_loss /= len(train_loader)
        accuracy = correct / total
        return train_loss,accuracy





