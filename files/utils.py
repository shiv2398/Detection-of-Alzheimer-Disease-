import yaml

def load_config():
    with open('files/data_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

DATA_CONFIG = load_config()

def load_train_config():
    with open('files/training_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

TRAIN_CONFIG = load_train_config()

def load_model_config():
    with open('files/model_configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

MODEL_CONFIG = load_model_config()

def freeze_layer(model):
    # Freeze the layers of the VGG16 model
    for param in model.parameters():
        param.requires_grad = False
    return model
