from wavenet_vocoder.upsample import UpSampleConv,ClariUpsampleConv
from train import get_data_loaders
from train import eval_model,load_checkpoint,build_model

def test_upsample():
    data_loaders = get_data_loaders('../data/ljspeech',-1)
    for phase, data_loader in data_loaders.items():
        train = (phase == "train")
        running_loss = 0.
        test_evaluated = False
        for step, (x, y, c, g, input_lengths) in enumerate(data_loader):
            c = c.unsqueeze(1)
            upconv1 = UpSampleConv()
            c1 =  upconv1(c)
            break



def test_sample():
    preste = '../presets/ljspeech_gaussian.json'
    model = build_model()