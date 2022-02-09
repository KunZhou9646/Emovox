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

### 2) Emotion training

## Step 4: Run-time conversion


## Training log
