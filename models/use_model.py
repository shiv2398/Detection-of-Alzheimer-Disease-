import os ,sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from files.utils import MODEL_CONFIG
from files.utils import TRAIN_CONFIG
from files.utils import DATA_CONFIG
from models import resnet,vgg_model,MyModel
def models_(name:str):    
    if name =='cnn':
        model_obj = MyModel(MODEL_CONFIG['resnet']['num_classes'])
        return model_obj
    elif name=='resnet18':
        model_obj=resnet.ResNet(MODEL_CONFIG['resnet']['resnet18']['depth'],
                             MODEL_CONFIG['resnet']['num_classes'])
        return model_obj
    elif name=='resnet34':
        model_obj=resnet.ResNet(MODEL_CONFIG['resnet']['resnet34']['depth'],
                             MODEL_CONFIG['resnet']['num_classes'])
        return model_obj
    elif name=='resnet52':
        model_obj=resnet.ResNet(MODEL_CONFIG['resnet']['resnet52']['depth'],
                             MODEL_CONFIG['resnet']['num_classes'])
        return model_obj
    elif name=='resnet101':
        model_obj=resnet.ResNet(MODEL_CONFIG['resnet']['resnet101']['depth'],
                             MODEL_CONFIG['resnet']['num_classes'])
        return model_obj
    elif name=='resnet152':
        model_obj=resnet.ResNet(MODEL_CONFIG['resnet']['resnet152']['depth'],
                             MODEL_CONFIG['resnet']['num_classes'])
        return model_obj 
    elif name=='resnet18':
        model_obj=resnet.ResNet(MODEL_CONFIG['resnet']['resnet18']['depth'],
                             MODEL_CONFIG['resnet']['num_classes'])
        return model_obj
    elif name=='resnet18':
         model_obj=resnet.ResNet(MODEL_CONFIG['resnet']['resnet18']['depth'],
                             MODEL_CONFIG['resnet']['num_classes'])
         return model_obj
    elif name=='default':
        model_obj=vgg_model(MODEL_CONFIG['vgg']['num_classes'])
        model=model_obj.model()
        return model
    else:
        pass
    