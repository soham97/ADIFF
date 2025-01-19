# ADIFF: Explaining audio difference using natural language

This repository hosts the Audio Difference Explanation datasets and ADIFF checkpoint. ADIFF is an audio prefix tuning-based language model with a cross-projection module and undergoes a three-step training process. ADIFF takes two audios and text prompt as input and produces different tiers of difference explanations as output. This involves identifying and describing audio events, acoustic scenes, signal characteristics, and their emotional impact on listeners.
![alt text](image.png)

## Setup
1. Install the required dependencies: `pip install -r requirements.txt`. For [conda](https://www.anaconda.com), run the following: 

```shell
cd adiff && \
conda create -n adiff python=3.8 && \
conda activate adiff && \
pip install -r requirements.txt
```

2. Download ADIFF weights: [Pretrained Model \[Anonymous drive\]](https://drive.google.com/file/d/1yixBwQcrH-Z59aqeG_YBRf9XcBInIXmV/view?usp=sharing)
3. Move the `adiff_base.pth` under `configs` folder

## Usage
The wrapper class allows easy interaction with the model. To use the wrapper, inputs required are:
- `config`: Choose between "base"
- `examples`: List of examples. Each example is a list containing three entries: audiopath1, audiopath2, prompt

Supported functions:
- `generate`: Produces text response for the given audio inputs and text prompt

### Generate difference explanation
```python
from wrapper import ADIFF

adiff = ADIFF(config="<choice of config>",model_path="<model weights")

examples = [
        ["<path1>", "<path2>", "explain the difference between the two audio in detail"],
        ["<path1>", "<path2>", "explain the difference between the two audio in one extended sentence"],
        ["<path1>", "<path2>", "explain the difference between the two audio in few words"],
    ]

response = adiff.generate(examples=examples, 
                            max_len=300, 
                            temperature=1.0
                            )
```

### Generate audio captions
```python
from wrapper import ADIFF

adiff = ADIFF(config="<choice of config>",model_path="<model weights")

examples = [
        ["<path1>", "<path2>", "caption the first audio"],
        ["<path1>", "<path2>", "caption the second audio"],
        ["<path1>", "<path2>", "caption both the audios"],
    ]

response = adiff.generate(examples=examples, 
                            max_len=300, 
                            temperature=1.0
                            )
```

### Dataset
The ACD and CLD dataset sources audio files from Clotho and AudioCaps dataset.  For the three tiers of difference annotation, the .csv are located under `data` folder

    .
    ├── ...
    ├── data              
    │   ├── ACD         # AudioCaps Difference Explanation
    |       ├── acd_test_adiff_fewwords_answer.csv
    |       ├── acd_test_adiff_sentence_answer.csv
    |       ├── acd_test_adiff_detail_answer.csv
    |       ├── ...
    │   ├── CLD         # Clotho Difference Explanation
    |       ├── cld_evaluation_adiff_fewwords_answer.csv
    |       ├── cld_evaluation_adiff_sentence_answer.csv
    |       ├── cld_evaluation_adiff_detail_answer.csv
    |       ├── ...
    └── ...
The audio files can be downloaded from their respective hosting website: [Clotho](https://zenodo.org/records/4783391) and [AudioCaps](https://github.com/cdjkim/audiocaps).

## Citation
```
@inproceedings{
    anonymous2024adiff,
    title={{ADIFF}: Explaining audio difference using natural language},
    author={Anonymous},
    booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=l4fMj4Vnly},
    note={under review}
}
```
