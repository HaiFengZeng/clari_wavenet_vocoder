from wavenet_vocoder.upsample import UpSampleConv,ClariUpsampleConv
from train import get_data_loaders

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
            upconv2 = ClariUpsampleConv()
            c2 = upconv2(c)
            print(c1 == c2)