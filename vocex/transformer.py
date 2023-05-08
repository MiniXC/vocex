import copy
import math

import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.dtype)
        return self.dropout(x)

# from https://pytorch.org/docs/1.13/_modules/torch/nn/modules/transformer.html#TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        return_additional_layer=None,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_additional_layer = return_additional_layer

    def forward(self, src, mask=None, src_key_padding_mask=None, condition=None):
            if src_key_padding_mask is not None:
                _skpm_dtype = src_key_padding_mask.dtype
                if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                    raise AssertionError("only bool and floating types of key_padding_mask are supported")
            
            output = src
            src_key_padding_mask_for_layers = src_key_padding_mask

            output_for_return = None

            for i, mod in enumerate(self.layers):
                if condition is not None:
                    output = output + condition
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)
                if self.return_additional_layer is not None and i == self.return_additional_layer:
                    output_for_return = output

            if self.norm is not None:
                output = self.norm(output)

            if output_for_return is not None:
                return output, output_for_return
            else:
                return output