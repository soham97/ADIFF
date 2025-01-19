import sys
sys.path.append('')
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel
import os

from model.audio import get_audio_encoder
from model.decoder import get_decoder

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        """Initialize a Batchnorm layer. """
        m.bias.data.fill_(0.)
        m.weight.data.fill_(1.)

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.linear1)
        init_layer(self.linear2)
        init_bn(self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, audioenc_name:str, d_in: int, d_out: int, sample_rate: int, window_size: int,
            hop_size: int, mel_bins: int, fmin: int, fmax: int, classes_num: int,) -> None:
        super().__init__()

        audio_encoder, pretrained_emb_size = get_audio_encoder(audioenc_name)

        self.base = audio_encoder(sample_rate, window_size,
            hop_size, mel_bins, fmin, fmax,
            classes_num, d_in)

        self.projection = Projection(pretrained_emb_size, d_out)

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output, out_dict

class ADiff(nn.Module):
    def __init__(self,
                # audio
                audioenc_name: str,
                sample_rate: int, 
                window_size: int, 
                hop_size: int, 
                mel_bins: int, 
                fmin: int, 
                fmax: int, 
                classes_num: int, 
                out_emb: int,
                # text decoder
                text_decoder: str,
                prefix_length: int,
                clip_length: int,
                prefix_size: int,
                num_layers: int,
                normalize_prefix: bool,
                d_proj: int,
                ):
        super().__init__()
        
        self.audio_encoder = AudioEncoder(
            audioenc_name, out_emb, d_proj,
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num,)

        self.caption_decoder = get_decoder('Decoder')(
            text_decoder, prefix_length, clip_length, prefix_size,
            num_layers, normalize_prefix,
        )

    def forward(self, input_dict):
        audio1 = input_dict['audio1']
        audio2 = input_dict['audio2']
        texts_enc = input_dict['input']
        texts_dec = input_dict['answer']

        audio_embed1, _, _ = self.audio_encoder(audio1)
        audio_embed2, _, _ = self.audio_encoder(audio2)
        caption_embed = self.caption_decoder.gpt.transformer.wte(texts_enc['input_ids'])

        out = self.caption_decoder(audio_embed1, audio_embed2, caption_embed, texts_dec)
        return out
    
    def generate_prefix_inference(self, input_dict):
        audio1 = input_dict['audio1']
        audio2 = input_dict['audio2']
        texts_enc = input_dict['input']

        audio_embed1, _, od1 = self.audio_encoder(audio1)
        audio_embed2, _, od2 = self.audio_encoder(audio2)
        caption_embed = self.caption_decoder.gpt.transformer.wte(texts_enc['input_ids'])
        prefix = self.caption_decoder.generate_prefix_inference(audio_embed1, audio_embed2, caption_embed)
        return prefix, od1, od2