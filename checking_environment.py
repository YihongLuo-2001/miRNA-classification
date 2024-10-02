try:
    import torch

    print('CUDA is available!' if torch.cuda.is_available() else 'Warning: CUDA is not available!')
except:
    print('Error: torch is not available!')
try:
    import sklearn

    print('sklearn is available!')
except:
    print('Error: sklearn is not available!')

