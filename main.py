
import torch
from training import early_stopping,train,valid
from data_preparation.class_loader import Simple_Dataset
from data_preparation.Sampler_dataset import Weight_samplerDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training.valid import validate
from models import MyModel
import pandas as pd 
import os ,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from files.utils import MODEL_CONFIG
from files.utils import TRAIN_CONFIG
from files.utils import DATA_CONFIG
from models import resnet,vgg_model,cnn
import argparse
import logging
from models.use_model import models_
# Create a logger instance
logger = logging.getLogger(__name__)

def training_(model_name=None,epochs:int=None):
    if epochs==None:
        epochs=TRAIN_CONFIG['training']['num_epochs']
    learning_rate=TRAIN_CONFIG['training']['learning_rate']

    #factor=TRAIN_CONFIG['training']['schedular']['factor']

    patience=TRAIN_CONFIG['training']['early_stopping']['patience']

    if model_name==None:
        model=models_('default')
    else:
        model=models_(model_name)

    if DATA_CONFIG['data_ratio']['unbalanced']:
        wt_obj=Weight_samplerDataset()
        train_loader,val_loadeer,test_loader=wt_obj.data_loaders()
    else:
        simple_obj=Simple_Dataset()
        train_loader,val_loader,test_loader=simple_obj.data_loaders()

    if TRAIN_CONFIG['training']['optimizer2']['name'] == 'Adam' and TRAIN_CONFIG['training']['scheduler']=='RLR':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #weight_decay=weight_decay_l2)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=TRAIN_CONFIG['training']['factor'], patience=patience, verbose=True)
    else:
        raise ValueError('Invalid optimizer specified in the configuration.')


    criterion = torch.nn.CrossEntropyLoss()
    early_stopping1 = early_stopping.EarlyStopping()

    if TRAIN_CONFIG['training']['metrics']['save']:
        train_losses, train_accs, train_precisions, train_recalls, train_f1s = [], [], [], [], []

    if TRAIN_CONFIG['validation']['metrics']['save']:
        val_losses, val_accs, val_precisions, val_recalls, val_f1s = [], [], [], [], []

    train_metrics_list = []
    val_metrics_list = []

    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')
        
        # Perform training and validation
        if TRAIN_CONFIG['training']['metrics']['save']:

            train_metrics = train.train(model, optimizer, criterion, train_loader)
            # Append metrics to the lists
            train_metrics_list.append(train_metrics)

        else:

            train_loss,train_accuracy=train.train(model, optimizer, criterion, train_loader)

        if TRAIN_CONFIG['validation']['metrics']['save']:
            val_metrics = valid.validate(model, criterion, val_loader)
            val_metrics_list.append(val_metrics)
            # Update the learning rate scheduler and early stopping
            val_loss = val_metrics['Val Loss']
        else:

            val_loss,val_accuracy=valid.validate(model, criterion, val_loader)
        scheduler.step(val_loss)
        
        if TRAIN_CONFIG['training']['early_stopping']['use']:
            early_stopping1(val_loss, model)
            
            # Check if early stopping criteria is met
            if early_stopping1.early_stop:
                print("Early stopping")
                break

        # Evaluate the model on test data
        if TRAIN_CONFIG['training']['inference']:
            print('Training Metrics:')
            print(f'Train Loss: {train_metrics["Train Loss"]:.4f} \
                Train Accuracy: {train_metrics["Accuracy"]:.4f} \
                Train Precision: {train_metrics["Precision"]:.4f} \
                Train Recall: {train_metrics["Recall"]:.4f} \
                Train F1 Score: {train_metrics["F1 Score"]:.4f}')
        
        if TRAIN_CONFIG['validation']['inference']:
            print('Validation Metrics:')
            print(f'Val Loss: {val_metrics["Val Loss"]:.4f} \
                Val Accuracy: {val_metrics["Accuracy"]:.4f} \
                Val Precision: {val_metrics["Precision"]:.4f} \
                Val Recall: {val_metrics["Recall"]:.4f} \
                Val F1 Score: {val_metrics["F1 Score"]:.4f}')

    # Convert the metrics lists to DataFrames
    train_metric_df = pd.DataFrame(train_metrics_list)
    val_metric_df = pd.DataFrame(val_metrics_list)

    # Save the metrics to CSV files
    if TRAIN_CONFIG['training']['metrics']['save']:
        #train_metric_df.to_csv('training_metrics.csv', index=False)
        pass

    if TRAIN_CONFIG['validation']['metrics']['save']:
        #val_metric_df.to_csv('validation_metrics.csv', index=False)
        pass

def testing(model_name,model_data_path):
    clss_ob=Simple_Dataset()
    test_loader=clss_ob.create_test_data()

    learning_rate=TRAIN_CONFIG['training']['learning_rate']
    if model_name==None:
        model=models_('default')
    else:
        model=models_(model_name)
    if TRAIN_CONFIG['training']['optimizer2']['name'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #weight_decay=weight_decay_l2)
    else:
        raise ValueError('Invalid optimizer specified in the configuration.')
    
    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = testing.test(model, test_loader, criterion)
    # Create a pandas dataframe from the lists of metrics
    metric_df=pd.DataFrame()
    if TRAIN_CONFIG['testing']['inference']:
        print('Validation Metrics:')
        print(f'Val Loss: {test_metrics["val_loss"]:.4f} \
                Val Accuracy: {test_metrics["accuracy"]:.4f} \
                Val Precision: {test_metrics["precision"]:.4f} \
                Val Recall: {test_metrics["recall"]:.4f} \
                Val F1 Score: {test_metrics["f1_score"]:.4f}')
        # create DataFrame for training and validation metrics

    if TRAIN_CONFIG['testing']['metrics']['save']:
        metric_df=pd.DataFrame(test_metrics,index=['testing'])
        metric_df.save('test_metrics.csv',index=False)


def main():
    # Configure the logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('Alzheimer Detection using Deep Learning')

    # Create the main parser
    parser = argparse.ArgumentParser(description='CLI description here')
    subparsers = parser.add_subparsers(dest='mode', help='Choose mode: train, test, or model_testing')

    # Create the parser for the train mode
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--train_data_path', help='Specify the training data path')
    train_parser.add_argument('--Epochs', help='Specify the training epochs')
    train_parser.add_argument('--model', help='Specify the model')
    train_parser.add_argument('--test_data_path', help='Specify the test data path')
    train_parser.add_argument('--val_exp', action='store_true', help='Use explicit validation data')
    train_parser.add_argument('--val_data_path', help='Explicit validation data path')
    train_parser.add_argument('--biased', action='store_true', help='Data is imbalanced')

    # Create the parser for the model_testing mode
    model_testing_parser = subparsers.add_parser('test', help='Saved model Testing')
    model_testing_parser.add_argument('--model',help='Specify the  model')
    model_testing_parser.add_argument('--test_data_path', help='Specify the test data path')
    model_testing_parser.add_argument('--model_path', help='Specify the model')
    #model_testing_parser.add_argument('--test_metric', help='Specify the test metric')
    #model_testing_parser.add_argument('--metric_save', action='store_true', help='Save metrics to file')

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.mode == 'train':
        # Check if the provided paths exist
        if not os.path.exists(args.train_data_path):
            print('Training data path does not exist.')
            return
        if not os.path.exists(args.test_data_path):
            print('Test data path does not exist.')
            return
        if args.val_exp and not os.path.exists(args.val_data_path):
            print('Validation data path does not exist.')
            return
        # Training with provided validation data
        DATA_CONFIG['data']['train_path'] = args.train_data_path
        DATA_CONFIG['data']['test_path'] = args.test_data_path
        if args.val_exp==False:
            DATA_CONFIG['data']['val_path'] = args.val_data_path
        if args.biased:
            DATA_CONFIG['data_ratio'] = True
        training_(args.model, int(args.Epochs))

    elif args.mode == 'test':
        DATA_CONFIG['data']['test_data']=args.test_data_path

        testing(args.model,args.model_path)  
    else:
        print("Invalid mode. Please choose 'train', 'test', or 'model_testing'.")

if __name__ == '__main__':
    main()

     