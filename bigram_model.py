from argparse import ArgumentParser
import struct
from functools import lru_cache
from itertools import accumulate
import time
import os
import numpy as np
import torch
from tqdm import tqdm

GPTNEOX_TOKENIZER_SIZE=50276
DATASET_PATH="/share/edc/home/alon_albalak/data/pile/preprocessed/{}/{}/{}_text_document"

def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)

dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}

def code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)

def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass
        
def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"

class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    # Write Magic string so we can check the file format then opening it again.
                    self._file.write(cls._HDR_MAGIC)
                    # Write version number
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", 1))
                    # Little endian unsigned 8 Bit integer
                    self._file.write(struct.pack("<B", code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    pointers = np.zeros(len(sizes), dtype=np.int64)
                    sizes = np.array(sizes, dtype=np.int64)

                    np.cumsum(sizes[:-1], out=pointers[1:])
                    pointers = pointers * dtype().itemsize
                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(sizes)))
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                # Little endian unsigned 64 Bit integer
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                # Little endian unsigned 8 Bit integer
                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
            )
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr
            )
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr
        )
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )
    
class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=True,
        use_shared_fs=True,
        max_samples = None,
        name_passthrough = False
    ):

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.data_prefix=data_prefix
        self.documents=documents
        self.seq_length=seq_length
        self.seed=seed
        self.use_shared_fs=use_shared_fs
        self.max_samples = max_samples
        self.name_passthrough = name_passthrough
        if num_samples is None:
            self._repeatable=True
            self._completed_epochs = 0


        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        if build_index_mappings:
            # Build index mappings.
            self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings_single_epoch(
                self.name,
                self.data_prefix,
                self.documents,
                self.indexed_dataset.sizes,
                self.seq_length,
                self.seed,
                self.use_shared_fs,
                self._completed_epochs
            )
            if self.max_samples is not None:
                if self.max_samples > self.shuffle_idx.shape[0] - 1:
                    print_rank_0(f"WARNING: max_samples ({self.max_samples}) is greater than the number of samples ({self.shuffle_idx.shape[0] - 1})")
                else:
                    print_rank_0(f"Resetting number of samples in {self.name} from {len(self.shuffle_idx)} to {self.max_samples}")
                    self.shuffle_idx = self.shuffle_idx[:self.max_samples+1]
            self.shuffle_idx_len = self.shuffle_idx.shape[0] - 1
            self.sample_idx_len = self.sample_idx.shape[0] - 1

            if self.shuffle_idx_len != self.sample_idx_len:
                print_rank_0(
                    f"WARNING: shuffle index length ({self.shuffle_idx_len}) is not equal to sample index length ({self.sample_idx_len})"
                )

    def __len__(self):
        return min(self.shuffle_idx_len, self.sample_idx_len)

    def __getitem__(self, idx):
        try:
            # Get the shuffled index.
            idx = self.shuffle_idx[idx]
            # Start and end documents and offsets.
            doc_index_f = self.sample_idx[idx][0]
            doc_index_l = self.sample_idx[idx + 1][0]
            offset_f = self.sample_idx[idx][1]
            offset_l = self.sample_idx[idx + 1][1]
            # If we are within the same document, just extract the chunk.
            if doc_index_f == doc_index_l:
                sample = self.indexed_dataset.get(
                    self.doc_idx[doc_index_f],
                    offset=offset_f,
                    length=offset_l - offset_f + 1,
                )
            else:
                # Otherwise, get the rest of the initial document.
                sample_list = [
                    self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)
                ]
                # Loop over all in between documents and add the entire document.
                for i in range(doc_index_f + 1, doc_index_l):
                    sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
                # And finally add the relevant portion of last document.
                sample_list.append(
                    self.indexed_dataset.get(
                        self.doc_idx[doc_index_l], length=offset_l + 1
                    )
                )
                sample = np.concatenate(sample_list)

            if self.name_passthrough:
                return {"text": np.array(sample, dtype=np.int64), "dataset_name": self.name}
            else:
                return {"text": np.array(sample, dtype=np.int64)}
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]

def _build_index_mappings_single_epoch(
    name, data_prefix, documents, sizes, seq_length, seed, use_shared_fs=True, epoch=0
    ):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
         training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Filename of the index mappings.
    _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_ep{}".format(epoch)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    # Load mappings.
    start_time = time.time()
    print_rank_0(" > loading doc-idx mapping from {}".format(doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading sample-idx mapping from {}".format(sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading shuffle-idx mapping from {}".format(shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(
        "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time)
    )
    print_rank_0("    total number of samples: {}".format(sample_idx.shape[0]))

    return doc_idx, sample_idx, shuffle_idx

def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def get_the_dataset(
    data_prefix,
    name,
    num_samples=None,
    seq_length=1024,
    seed=42,
    skip_warmup=True,
    build_index_mappings=True,
    max_samples=None,
    name_passthrough=False,
    ):

    indexed_dataset = MMapIndexedDataset(data_prefix, skip_warmup=skip_warmup)
    total_num_of_documents = indexed_dataset.sizes.shape[0]
    print_rank_0("    {}:".format(name))
    print_rank_0("     no. of documents:{}".format(total_num_of_documents))
    dataset = None
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
    dataset = GPT2Dataset(
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=build_index_mappings,
        max_samples=max_samples,
        name_passthrough=name_passthrough
    )
    return dataset

class bigram:
    def __init__(self, vocab_size):
        self.counts = np.ones((vocab_size+1, vocab_size+1), dtype=np.uint32)
        self.vocab_size = vocab_size
        self.total_count = vocab_size ** 2
        
    def update(self, x):
        self.counts[x[:-1], x[1:]] += 1
        self.total_count += len(x)

    def get_prob(self, x):
        return self.counts[x[:-1], x[1:]] / self.total_count
    
    def get_log_prob(self, x):
        return np.log(self.get_prob(x))
    
    def get_perplexity(self, x):
        return np.exp(-np.sum(self.get_log_prob(x)) / len(x))
    
    def get_entropy(self, x):
        return -np.sum(self.get_log_prob(x)) / len(x)

def train_bigram(dataset_name, num_samples):
    print("**** Loading dataset...")
    split="train"
    data_path = DATASET_PATH.format(split, dataset_name, dataset_name)
    indexed_dataset = get_the_dataset(data_path, f"{split}_{dataset_name}")
    print(f"**** Dataset disk size: {len(indexed_dataset)*1025*8/1024/1024/1024} GB")

    if len(indexed_dataset) < num_samples:
        print(f"**** Dataset has only {len(indexed_dataset)} samples, training on all of them")
        num_samples = len(indexed_dataset)
    
    print(f"**** Creating bigram model...")
    start_time = time.time()
    bigram_model = bigram(GPTNEOX_TOKENIZER_SIZE)
    print(f"**** Created bigram model in {time.time() - start_time} seconds")

    print("**** Training bigram model...")
    for i in tqdm(range(num_samples)):
        x = indexed_dataset[i]["text"]
        bigram_model.update(x)
    print("**** Trained bigram model in {:3.3f} seconds".format(time.time() - start_time))
    return bigram_model

def evaluate_bigram(bigram_model, dataset_name, num_samples, split):
    print(f"**** Loading evaluation split: {split}...")
    data_path = DATASET_PATH.format(split, dataset_name, dataset_name)
    if split == "validation":
        indexed_dataset = get_the_dataset(data_path, f"valid_{dataset_name}")
    else:
        indexed_dataset = get_the_dataset(data_path, f"{split}_{dataset_name}")
    print(f"**** Dataset disk size: {len(indexed_dataset)*1025*8/1024/1024/1024} GB")
    
    if len(indexed_dataset) < num_samples:
        print(f"**** Dataset has only {len(indexed_dataset)} samples, evaluating all of them")
        num_samples = len(indexed_dataset)

    print("**** Evaluating bigram model...")
    start_time = time.time()
    lm_loss = 0
    perplexity = 0
    for i in tqdm(range(num_samples)):
        x = indexed_dataset[i]["text"]
        lm_loss += bigram_model.get_entropy(x)
        perplexity += bigram_model.get_perplexity(x)
    print("**** Evaluated bigram model in {:3.3f} seconds".format(time.time() - start_time))
    return lm_loss / num_samples, perplexity / num_samples

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="/share/edc/home/alon_albalak/bigram_models",
        help="The path to save the bigram model",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to train on",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        required=True,
        help="The number of samples to train on",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a bigram model",
    )
    parser.add_argument(
        "--evaluate_samples",
        type=int,
        default=10000000,
        help="The number of samples to evaluate on",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate a bigram model",
    )
    args = parser.parse_args()
    save_path = args.save_path
    dataset_name = args.dataset_name
    train_samples = args.train_samples
    evaluate_samples = args.evaluate_samples
    train = args.train
    evaluate = args.evaluate
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if train:
        if not os.path.exists(os.path.join(save_path,f"{dataset_name}_bigram_model.npy")):
            bigram_model = train_bigram(dataset_name, train_samples)
            print(f"**** Saving bigram model...")
            np.save(os.path.join(save_path,f"{dataset_name}_bigram_model.npy"), bigram_model.counts)
            print(f"**** Saved bigram model")
        else:
            print(f"**** Bigram model already exists in {save_path}, skipping training")
    if evaluate:
        bigram_model = bigram(GPTNEOX_TOKENIZER_SIZE)
        bigram_model.counts = np.load(os.path.join(save_path,f"{dataset_name}_bigram_model.npy"))
        print("Evaluation results:")
        for split in ["train", "validation", "test"]:
            lm_loss, perplexity = evaluate_bigram(bigram_model, dataset_name, evaluate_samples, split)
            print(f"lm_loss {split}: {lm_loss}")
            print(f"perplexity {split}: {perplexity}")


if __name__ == "__main__":
    main()