import warnings
warnings.filterwarnings("ignore")
import numpy as np
from transformers import AutoTokenizer
import os
import torch
from collections import OrderedDict
from importlib_resources import files
import yaml
import argparse
import torchaudio
import torchaudio.transforms as T
import collections
import random
from model.model import get_model_class
from tqdm import tqdm
import torch.nn.functional as F
import random

class ADIFF():
    """
    A class for interfacing ADIFF model.
    """
    def __init__(self, config_path, model_path, use_cuda=True):
        self.file_path = os.path.realpath(__file__)
        self.model_path = files('config').joinpath(model_path)
        self.config_path = files('config').joinpath(config_path)
        self.use_cuda = use_cuda
        self.model, self.tokenizer, self.args = self.get_model_and_tokenizer(config_path=self.config_path)
        self.model.eval()

    def read_config_as_args(self,config_path):
        return_dict = {}
        with open(config_path, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            return_dict[k] = v
        return argparse.Namespace(**return_dict)

    def get_model_and_tokenizer(self, config_path):
        r"""Load Adiff with args from config file"""
        args = self.read_config_as_args(config_path)
        args.model["decoder"]["prefix_dim"] = args.model["encoder"]["d_proj"]
        args.model["decoder"]["total_prefix_length"] = 3*args.model["decoder"]["prefix_length"] + 1

        Model = get_model_class(model_type=args.model['model_type'])
        model = Model(
            # audio
            audioenc_name = args.model['encoder']['audioenc_name'],
            sample_rate = args.data['sampling_rate'],
            window_size = args.model['encoder']['window_size'],
            hop_size = args.model['encoder']['hop_size'],
            mel_bins = args.model['encoder']['mel_bins'],
            fmin = args.model['encoder']['fmin'],
            fmax = args.model['encoder']['fmax'],
            classes_num = args.model['encoder']['num_classes'],
            out_emb = args.model['encoder']['out_emb'],
            # text decoder
            text_decoder = args.model['decoder']['text_decoder'],
            prefix_length = args.model['decoder']['prefix_length'],
            clip_length = args.model['decoder']['prefix_length_clip'],
            prefix_size = args.model['decoder']['prefix_dim'],
            num_layers = args.model['decoder']['num_layers'],
            normalize_prefix = args.model['decoder']['normalize_prefix'],
            d_proj = args.model['encoder']['d_proj'],
        )
        model_state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(model_state_dict)
        except:
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:] # remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        tokenizer = AutoTokenizer.from_pretrained(args.model["decoder"]["text_decoder"])
        if 'gpt' in args.model["decoder"]["text_decoder"]:
            tokenizer.add_special_tokens({'pad_token': '!'})

        if self.use_cuda and torch.cuda.is_available():
            model = model.cuda()
        
        return model, tokenizer, args

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

    def load_audio_into_tensor(self, audio_path, audio_duration, resample=True):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        resample_rate = self.args.data["sampling_rate"]
        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        audio_time_series = audio_time_series.reshape(-1)
        sample_rate = resample_rate

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration*sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  audio_duration*sample_rate]
        return torch.FloatTensor(audio_time_series)

    def preprocess_audio(self, audio_files, resample):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        for audio_file in audio_files:
            audio_tensor = self.load_audio_into_tensor(
                audio_file, self.args.data["segment_seconds"], resample)
            audio_tensor = audio_tensor.reshape(
                1, -1).cuda() if self.use_cuda and torch.cuda.is_available() else audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return self.default_collate(audio_tensors)

    def preprocess_text(self, prompts, add_text):
        r"""Load list of prompts and return tokenized text"""
        tokenized_texts = []
        for ttext in prompts:
            if add_text:
                tok = self.dec_tokenizer.encode_plus(text=ttext, add_special_tokens=True, return_tensors="pt")
            else:
                ttext = ttext + ' <|endoftext|>' if 'gpt' in self.args.model["decoder"]["text_decoder"] else ttext
                tok = self.tokenizer.encode_plus(
                            text=ttext, add_special_tokens=True,\
                            max_length=self.args.data["ip_text_len"], 
                            pad_to_max_length=True, return_tensors="pt")
                
            for key in tok.keys():
                tok[key] = tok[key].reshape(-1).cuda() if self.use_cuda and torch.cuda.is_available() else tok[key].reshape(-1)
            tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)

    def _generate_greedy(self, embed=None,entry_length=300, temperature=1., 
            stop_token: str = ' <|endoftext|>',entry_count=1,top_p=0.8,):
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        tokens = None
        generated_list = []
        filter_value = -float("Inf")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for entry_idx in range(entry_count):
                if embed is not None:
                    generated = embed
                else:
                    if tokens is None:
                        tokens = torch.tensor(self.tokenizer.encode(prompt))
                        tokens = tokens.unsqueeze(0).to(device)
                    generated = self.model.gpt.transformer.wte(tokens)

            for i in tqdm(range(entry_length)):
                outputs = self.model.caption_decoder.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = self.model.caption_decoder.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = self.tokenizer.decode(output_list)
            generated_list.append(output_text)

        return [generated_list]
    
    def generate(self, examples, max_len, temperature, stop_token=' <|endoftext|>', audio_resample=True):
        r"""Produces text response for the given audio file and text prompts
        examples: (list<list>) List of examples. Each example is a list containing three entries [audio path 1, audio path 2, text prompt]
        max_len: (int) maximum length for text generation. Necessary to stop generation if GPT2 gets "stuck" producing same token
        temperature: (float) temperature parameter for GPT2 generation
        stop_token: (str) token used to stop text generation 
        audio_resample (bool) True for resampling audio. The model support only 32 kHz
        """
        preds = []
        for example in examples:
            audio_path1, audio_path2, text_prompt = example
        
            audio1, audio2 = self.preprocess_audio([audio_path1,audio_path2], resample=audio_resample)
            textprompt = self.preprocess_text([text_prompt], add_text=False)
            d = {
                    "audio1": audio1,
                    "audio2": audio2,
                    "input": textprompt,
                }
            prefix, _, _ = self.model.generate_prefix_inference(d)
            pred = self._generate_greedy(embed=prefix, temperature=temperature, stop_token=stop_token, entry_length=max_len)
            preds.append(pred[0][0])

        return preds
