from pathlib import Path
import gzip

import torch
import onnx
import onnxruntime as ort
import numpy as np
from onnxruntime.quantization import quantize_dynamic, quantize_static
from onnxconverter_common import float16

from vocex import Vocex, OnnxVocexWrapper

def export_onnx(checkpoint, onnx_path):
    vocex = Vocex.from_pretrained(checkpoint)
    model = vocex.model
    model.onnx_export = True

    dummy_input = torch.randn(1, 512, 80)
    torch.onnx.export(model, dummy_input, onnx_path, verbose=False)

def export_wrapped_onnx(checkpoint, onnx_path):
    vocex = OnnxVocexWrapper.from_pretrained(checkpoint)
    dummy_input = torch.randn(512*256)
    torch.onnx.export(vocex, dummy_input, onnx_path, verbose=False)

def compress_onnx(checkpoint, onnx_path):
    onnx_path = Path(onnx_path)
    quantized_path = onnx_path.with_suffix('.int8.onnx')
    export_onnx(checkpoint, onnx_path.with_suffix('.tmp.onnx'))
    quantize_dynamic(
        onnx_path,
        quantized_path,
        op_types_to_quantize=['MatMul', 'Gemm', 'QLinearConv', 'QLinearMatMul']
    )
    float_16_path = onnx_path.with_suffix('.float16.onnx')
    onnx_model = onnx.load(onnx_path.with_suffix('.tmp.onnx'))
    onnx_model = float16.convert_float_to_float16(
        onnx_model,
        min_positive_val=1e-7,
        max_finite_val=1e4,
        keep_io_types=False,
        disable_shape_infer=False,
        op_block_list=float16.DEFAULT_OP_BLOCK_LIST,
        node_block_list=None
    )
    onnx.save(onnx_model, float_16_path)
    # delete the tmp file
    onnx_path.with_suffix('.tmp.onnx').unlink()

def check_onnx(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

def check_onnxruntime(checkpoint, onnx_path, use_float16=False):
    # decompress the onnx file
    sess = ort.InferenceSession(onnx_path)
    random_data = torch.randn(1, 512, 80)
    if use_float16:
        random_data = random_data.to(dtype=torch.float16)
    outputs = sess.run(None, {'mel': random_data.numpy()})
    # compare the output with pytorch
    model = Vocex.from_pretrained(checkpoint).model
    model.eval()
    model.onnx_export = True
    with torch.no_grad():
        if use_float16:
            random_data = random_data.to(dtype=torch.float32)
        torch_outputs = [o.numpy() for o in model(random_data)]
    for i, (output, torch_output) in enumerate(zip(outputs, torch_outputs)):
        output = output / output.max()
        torch_output = torch_output / torch_output.max()
        print(f'output {i} max diff: {np.abs(output - torch_output).max()}')


def main():
    checkpoint = Path('models/checkpoint_full.ckpt')
    checkpoint_half = Path('models/checkpoint_half.ckpt')
    onnx_path = Path('models/onnx/model.onnx')
    print("exporting onnx")
    export_onnx(checkpoint, onnx_path)
    print("compressing onnx")
    compress_onnx(checkpoint_half, onnx_path)
    print("checking onnx")
    check_onnx(onnx_path)
    check_onnxruntime(checkpoint, onnx_path)
    print("checking fp16 onnx")
    check_onnxruntime(checkpoint, onnx_path.with_suffix('.float16.onnx'), use_float16=True)
    print("checking int8 onnx")
    check_onnxruntime(checkpoint, onnx_path.with_suffix('.int8.onnx'))
    print("exporting wrapped onnx")
    export_wrapped_onnx(checkpoint, onnx_path.with_suffix('.wrapped.onnx'))

if __name__ == '__main__':
    main()