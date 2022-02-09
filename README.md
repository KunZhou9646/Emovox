# Emovox
This is the implementation of the paper "Emotion Intensity and its Control for Emotional Voice Conversion".

![image info](./stage3_update.png)

## Database:
We use ESD database, which is an emotional speech database that can be downloaded here: https://hltsingapore.github.io/ESD/. In this paper, we choose "0013" to perform all the experiments. To run the codes, you first need to customize your data path correctly, and generate phoneme transcriptions with Festival. More details can be found in https://github.com/jxzhanggg/nonparaSeq2seqVC_code.


## Step 1: Learning relative attributes

### 1) Extracting open-simle features

```Bash
python pre-processing.py
```

### 2) Training relative ranking function

```Matlab
main.m
```

## Step 2: Emotion recognizer training

```Bash
python train_ser.py
```

## Step 3: Emovox training

### 1) Style Pre-training

You need to download VCTK corpus and customize it accordingly, and then perform feature extraction:
```Bash
$ cd reader
$ python extract_features.py (please customize "path" and "kind", and edit the codes for "spec" or "mel-spec")
$ python generate_list_mel.py
```

The pre-training procedure is same as the pretraining in  https://github.com/jxzhanggg/nonparaSeq2seqVC_code. You can download the pre-trained models from Stage I: Style Initialization here: https://drive.google.com/file/d/1oqk-PSREwpFNTyeREwcUry13WZ1LYl6U/view?usp=sharing. With the released pre-trained models, you can directly perform Stage II: Emotion Training. If you would like to pre-train it by yourself, you can try the following:
```Bash
$ python train.py -l logdir \
-o outdir --n_gpus=1 --hparams=speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.
```

### 2) Emotion training

You need to download ESD corpus and customize it accordingly, and then perform feature extraction:
```Bash
$ cd reader
$ python extract.py (please customize "path" and "kind", and edit the codes for "spec" or "mel-spec")
$ python generate_list_mel.py
```

```Bash
$ python train.py -l logdir \
-o outdir_emotion_IS --n_gpus=1 -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir/checkpoint_234000 (The path to your Pre-trained models from Stage I)' --warm_start
```

## Step 4: Run-time conversion

(1) Generate emotion embedding from the emotion encoder:

Please remember to customize the paths in hparam.py...
```Bash
$ cd conversion
$ python inference_embedding.py -c '/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/outdir_emotion_update/checkpoint_3200 [YOUR EMOTION TRAINING CHECKPOINT]' --hparams speaker_A='Neutral',speaker_B='Happy',speaker_C='Sad',speaker_D='Angry',training_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/testing_mel_list.txt',SC_kernel_size=1
```
(2) Convert the source speech to the target emotion: [FOR EXAMPLE: convert emotion D to emotion A]
```Bash
$ cd conversion
$ python inference_A.py -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir_emotion_update/checkpoint_3200[YOUR EMOTION TRAINING CHECKPOINT]' --num 20 --hparams validation_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/evaluation_mel_list.txt',SC_kernel_size=1
```
Please customize inference.py to generate your intended emotion type.


## Training log
