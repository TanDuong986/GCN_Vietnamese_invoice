from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import timeit

config = Cfg.load_config_from_name('vgg_seq2seq')
config['cnn']['pretrained'] = True
config['device'] = 'cuda'
detector = Predictor(config)

def test_time():
    path = '/home/dtan/Documents/GCN/GCN_Vietnam/Code/detect_word/origin/ROI/sub_10.jpg'
    img = Image.open(path)
    s = detector.predict(img) #1.67 s without cuda | 0.02 with cuda

ext = timeit.timeit(test_time,number=100)/100
print(ext)