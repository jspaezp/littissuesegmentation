import numpy as np
import littissuesegmentation as lts

def test_chunk_generator():
    stride = 256
    chunk_size = 512
    img = np.ones((5001, 5001, 3))

    sample_chunker = lts.chunking.chunk_genrator(img, stride, chunk_size, verbose=False)
    for i, x in enumerate(sample_chunker):
        continue

    assert i > 0

    img = np.ones((5001, 5001))
    sample_chunker = lts.chunking.chunk_genrator(img, stride, chunk_size, verbose=False)
    for i, x in enumerate(sample_chunker):
        continue

    assert i > 0


def test_chunk_dataloader():
    stride = 256
    chunk_size = 512
    img = np.ones((5001, 5001, 3))

    cdl = lts.dataloaders.ImgChunkDataloader(img, stride, chunk_size, verbose=False)
    for b in cdl:
        break

    return b
