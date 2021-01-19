import math
import numpy as np
from tqdm.auto import tqdm


def calculate_out_size(img_shape, stride, chunk_size, verbose=False):
    assert stride <= chunk_size, (
        "stride should be smaller than the chunk size"
        " to have 100 percent coverage of the image"
    )
    # calculate the size of the output image
    size_out_x = (math.ceil((img_shape[0] - chunk_size) / stride) * stride) + chunk_size
    size_out_y = (math.ceil((img_shape[1] - chunk_size) / stride) * stride) + chunk_size
    if verbose:
        print(f"Expanding output image from {img_shape} to {size_out_x} {size_out_y}")

    return size_out_x, size_out_y


def chunk_position_genrator(img_shape, stride, chunk_size, verbose=False):
    size_out_x, size_out_y = calculate_out_size(
        img_shape, stride, chunk_size, verbose=verbose
    )

    num_steps_x = (size_out_x - chunk_size) / stride
    assert num_steps_x % 1 == 0
    num_steps_x = int(num_steps_x)
    num_steps_y = (size_out_y - chunk_size) / stride
    assert num_steps_y % 1 == 0
    num_steps_y = int(num_steps_y)

    with tqdm(total=(num_steps_y + 1) * (num_steps_x + 1)) as pbar:
        for xi in range(num_steps_x + 1):
            if verbose:
                print(f"X start: {xi * stride}, end: {(xi*stride) + chunk_size}")
            for yi in range(num_steps_y + 1):
                if verbose:
                    print(f"Y start: {yi * stride}, end: {(yi*stride) + chunk_size}")
                pbar.update(1)
                yield xi * stride, yi * stride


def chunk_genrator(img, stride, chunk_size, verbose=False):
    position_gen = chunk_position_genrator(img.shape, stride, chunk_size, verbose)
    extra_dims = img.shape[2:]
    extra_dims_pad = (0, 0) * len(extra_dims)

    for xi, yi in position_gen:
        chunk = img[xi : xi + chunk_size, yi : yi + chunk_size, ...]
        padding_tuple = (
            (0, chunk_size - chunk.shape[0]),
            (0, chunk_size - chunk.shape[1]),
        )

        if len(img.shape) > 2:
            padding_tuple += (extra_dims_pad,)
        chunk = np.pad(chunk, padding_tuple, "edge")

        assert all([x == chunk_size for x in chunk.shape[:2]])
        yield chunk, (xi, yi)
