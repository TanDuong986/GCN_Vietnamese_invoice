from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def gen_model():
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['cnn']['pretrained'] = True
    config['device'] = 'cuda'
    return Predictor(config)

