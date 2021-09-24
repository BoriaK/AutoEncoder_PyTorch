import torch
# import torchac

datasetFolder = r'../dataSet/DataSet1'


def PSNR(original, compressed):
    mse = torch.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


x = torch.randn(1, 64)
B = 12
y = torch.round((2 ** (B - 1)) * x)
# Encode to bytestream.
output_cdf = ...  # Get CDF from your model, shape B, C, H, W, Lp
# sym = ...  # Get the symbols to encode, shape B, C, H, W.
# byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)

# Number of bits taken by the stream
# real_bits = len(byte_stream) * 8
