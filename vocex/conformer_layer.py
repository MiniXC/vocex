from torch import nn
from torch.nn import TransformerEncoderLayer

from .utils import DepthwiseConv1d

class ConformerLayer(TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        old_kwargs = {k: v for k, v in kwargs.items() if "conv_" not in k}
        super().__init__(*args, **old_kwargs)
        del self.linear1
        del self.linear2
        if "conv_depthwise" in kwargs and kwargs["conv_depthwise"]:
            self.conv1 = DepthwiseConv1d(
                kwargs["conv_in"],
                kwargs["conv_filter_size"],
                kernel_size=kwargs["conv_kernel"][0],
                padding=(kwargs["conv_kernel"][0] - 1) // 2,
            )
            self.conv2 = DepthwiseConv1d(
                kwargs["conv_filter_size"],
                kwargs["conv_in"],
                kernel_size=kwargs["conv_kernel"][1],
                padding=(kwargs["conv_kernel"][1] - 1) // 2,
            )
        else:
            self.conv1 = nn.Conv1d(
                kwargs["conv_in"],
                kwargs["conv_filter_size"],
                kernel_size=kwargs["conv_kernel"][0],
                padding=(kwargs["conv_kernel"][0] - 1) // 2,
            )
            self.conv2 = nn.Conv1d(
                kwargs["conv_filter_size"],
                kwargs["conv_in"],
                kernel_size=kwargs["conv_kernel"][1],
                padding=(kwargs["conv_kernel"][1] - 1) // 2,
            )

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=False):
        x = src
        if self.norm_first:
            if not need_weights:
                attn = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            else:
                attn, weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, need_weights=need_weights)
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            if not need_weights:
                attn = self._sa_block(x, src_mask, src_key_padding_mask)
            else:
                attn, weights = self._sa_block(x, src_mask, src_key_padding_mask, need_weights=need_weights)
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))
        if need_weights:
            return x, weights
        else:
            return x

    def _ff_block(self, x):
        x = self.conv2(
            self.dropout(self.activation(self.conv1(x.transpose(1, 2))))
        ).transpose(1, 2)
        return self.dropout2(x)

    def _sa_block(
            self, 
            x,
            attn_mask,
            key_padding_mask=None,
            need_weights=False,
        ):
        if not need_weights:
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights)[0]
        else:
            x, weights = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights)
        if need_weights:
            return self.dropout1(x), weights
        else:
            return self.dropout1(x)