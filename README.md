# TTS project barebones


# done
rewrite LN in tranfsormer, check links in ipynb about it

delete val/eval
add paths from TrainConfig to configs
dont forget to add downloading stuff to test.py
add separate dropout for variance 
# TODO

delete tests
mb audio folder and utils from fastspeech isnt needed
Default batch_size=16

add calc_pitches to installation


# report 
changed mhead attention to pre norm, nothing changed
changed all other attentions to pre norm, didnt notice any changes

copied parts of code for waveglow and preprocessing from https://github.com/xcmyz/FastSpeech.git (same parts what were used in the seminar)

pitch without cwt because only one librrary has icwt, and it has almost no documentation
## Installation guide


```shell
pip install -r ./requirements.txt

wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt data/

gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p pretrained_model/
mkdir -p data/waveglow/
mv waveglow_256channels_ljs_v2.pt pretrained_model/waveglow_256channels.pt
gdown https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j
tar -xvf mel.tar.gz

wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip >> /dev/null
mv pretrained_model data/waveglow/
mv alignments data/
mv mels data/
rm mel.tar.gz
rm alignments.zip
rm LJSpeech-1.1.tar.bz2
```

```shell
python3 calc_energies.py
```
OR download archive from https://drive.google.com/file/d/1U-bBkdmKvIBjSx0T1O87YwIY-502YitC/view?usp=sharing and unarchive into the ./data folder

Download model from https://drive.google.com/file/d/19-HH878SpTFWugKdMeV4cf5f4PTyFrG3/view to default_test_model/checkpoint.pth i cant write a script for that in time.

To test model on clean test:
```shell
python3 test.py -c hw_asr/default_test_clean_config.json -r default_test_model/checkpoint.pth
```

To test model on other test:
```shell
python3 test.py -c hw_asr/default_test_other_config.json -r default_test_model/checkpoint.pth
```

To train the model run:

```shell
python3 train.py -c hw_asr/configs/train_run_1.json
python3 train.py -c hw_asr/configs/train_run_2.json -l <path to the model_best.pth from the last step>
python3 train.py -c hw_asr/configs/train_run_3.json -l <path to the model_best.pth from the last step>
python3 train.py -c hw_asr/configs/train_run_4.json -l <path to the model_best.pth from the last step>
python3 train.py -c hw_asr/configs/train_run_5.json -l <path to the model_best.pth from the last step>
python3 train.py -c hw_asr/configs/train_run_6.json -l <path to the model_best.pth from the last step>
python3 train.py -c hw_asr/configs/train_run_6.json -l <path to the model_best.pth from the last step>
python3 train.py -c hw_asr/configs/train_run_8.json -l <path to the model_best.pth from the last step>
```
Yes, you should use config train_run_6 twice, this is not a typo.


## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

