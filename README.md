# LLM training data querying through infini-gram mini

> **Note:** I am not the author of the original infini-gram mini engine or paper.
> This repository is a personal build on top of the original
> [infini-gram mini](https://infini-gram-mini.io/code) project
> ([paper](https://arxiv.org/abs/2506.12229), [project home](https://infini-gram-mini.io/)).
> All credit for the core engine and indexing pipeline goes to
> Hao Xu, Jiacheng Liu, Yejin Choi, Noah A. Smith, and Hannaneh Hajishirzi.

What this repo adds on top of the original:

- An **Apptainer container** definition for reproducible HPC deployment (`scripts/infini_gram_mini.def`)
- **SLURM job scripts** for building the container and running indexing jobs (`scripts/`)
- A **`pyproject.toml`** for installing the package locally with pip
- A **CLI query tool** (`api/query.py`) with count / search / find subcommands and formatted output
- A **REST API server** (`api/api_server.py`) for serving an index over HTTP

---

## Repository layout

```text
infini-gram-mini/
├── infini-gram-mini/              # Upstream source code
│   ├── engine/
│   │   └── src/
│   │       ├── cpp_engine.cpp     # C++ query backend (pybind11)
│   │       ├── cpp_engine.h
│   │       ├── engine.py          # Python wrapper
│   │       └── models.py
│   └── indexing/
│       ├── cpp/
│       │   └── indexing.cpp       # Compiled to cpp_indexing binary
│       ├── rust_indexing          # Compiled Rust binary (built from third_party/suffix_array)
│       └── indexing.py            # Core prepare / build_sa_bwt logic
├── scripts/
│   ├── build_apptainer.sh         # SLURM script: builds the Apptainer image
│   ├── infini_gram_mini.def       # Apptainer container definition
│   ├── index_v2_parquet.sh        # SLURM: single-job indexing
│   └── index_v2_parquet_array.sh  # SLURM: array-job indexing
├── third_party/
│   ├── nlohmann/                  # JSON header
│   ├── parallel_sdsl/             # SDSL + divsufsort (used by indexing)
│   ├── sdsl/                      # SDSL + divsufsort (used by query engine)
│   └── suffix_array/              # Rust source for rust_indexing binary
├── api/
│   ├── query.py                   # CLI query tool (count / search / find)
│   ├── api_server.py              # Flask REST API server
│   └── api_config.json            # Index config (edit this to point at your indexes)
├── pyproject.toml                 # Python package definition
```

---

## Installation

Two paths are supported: **plain Python** (compile everything natively on the host) or **Apptainer** (recommended for HPC — bundles all toolchains in one image).

---

### Option A — Python (bare metal)

Requirements: Python ≥ 3.10, GCC ≥ 11, Rust toolchain (`cargo`).

#### 1. Install Python dependencies

```bash
pip install pybind11 pyarrow numpy zstandard
```

Or install the package directly (this also records the dependencies):

```bash
pip install -e .
```

#### 2. Build the Rust suffix-array binary

```bash
cd third_party/suffix_array
cargo build --release
cp target/release/rust_indexing ../../infini-gram-mini/indexing/rust_indexing
```

#### 3. Compile the C++ indexing binary

```bash
cd infini-gram-mini/indexing
g++ -std=c++17 -O3 \
    cpp/indexing.cpp -o cpp/cpp_indexing \
    -I../../third_party/parallel_sdsl/include \
    -L../../third_party/parallel_sdsl/lib \
    -lsdsl -ldivsufsort -ldivsufsort64
```

#### 4. Compile the query engine (Python extension)

```bash
cd infini-gram-mini/engine
c++ -std=c++17 -O3 -shared -fPIC \
    $(python3 -m pybind11 --includes) \
    src/cpp_engine.cpp \
    -o src/cpp_engine$(python3-config --extension-suffix) \
    -I../../third_party/sdsl/include \
    -L../../third_party/sdsl/lib \
    -lsdsl -ldivsufsort -ldivsufsort64 -pthread
```

---

### Option B — Apptainer (recommended for HPC)

The container bundles Python 3.12, GCC, Rust, and all compiled binaries in one reproducible image (x86_64 / amd64 only). Build it once; run it on any compatible node.

#### 1. Build the image (submit as a SLURM job)

From the repo root:

```bash
sbatch scripts/build_apptainer.sh
```

The finished image is written to `<path>/infini_gram_mini.sif`.

#### 2. Run indexing

Via Python CLI:

```bash
python -m indexing.indexing \
    --data_dir /path/to/parquet_files \
    --save_dir /path/to/index_output
```

Via apptainer:

```bash
apptainer run /path/to/infini_gram_mini.sif \
    --data_dir /path/to/parquet_files \
    --save_dir /path/to/index_output
```

Or submit via SLURM:

```bash
sbatch scripts/index_parquet.sh        # single job
sbatch scripts/index_parquet_array.sh  # array job
```

#### 3. Query inside the container

Bind-mount the repo so the container can read your config and index:

```bash
apptainer exec \
    --bind /path/to/infini-gram-mini:/repo \
    --bind /path/to/index:/index \
    /path/to/infini_gram_mini.sif \
    python3 /repo/api/query.py \
        --config /repo/api/api_config.json \
        count "natural language processing"
```

---

## Querying an existing index

### 1. Edit the index config

`api/api_config.json` is a JSON array where each entry describes one named index.

`index_dirs` accepts two forms:
- **A root directory string** — all immediate subdirectories are loaded as shards (sorted). Convenient when the indexing job produces `save_dir/00/`, `save_dir/01/`, etc.
- **A list of directory strings** — explicit list of shard paths, loaded in the given order.

```json
[
    {
        "name": "my_index",
        "index_dirs": "/path/to/index",
        "load_to_ram": false,
        "get_metadata": true
    }
]
```

Or with an explicit shard list:

```json
[
    {
        "name": "my_index",
        "index_dirs": ["/path/to/index/00", "/path/to/index/01"],
        "load_to_ram": false,
        "get_metadata": true
    }
]
```

### 2. Query via CLI

`api/query.py` loads the engine directly — no server needed:

```bash
cd api

# Count occurrences
python query.py --config api_config.json count "natural language processing"
# Count:   83,470
# Latency: 12.3 ms

# Search and retrieve matching document contexts (query highlighted in output)
python query.py --config api_config.json search "transformer model"

# Target a specific named index
python query.py --config api_config.json --index my_index count "BERT"

# Customise result count and context window
python query.py --config api_config.json search "GPT" --max_results 5 --max_ctx_len 300

# Raw suffix-array intervals per shard
python query.py --config api_config.json find "natural language processing"
```

### 3. (Optional) REST API server

Start the server:

```bash
cd api
python api_server.py --config api_config.json --port 5000
```

Then query it from any client:

```bash
# via query.py
python query.py --api_url http://localhost:5000 count "natural language processing"

# via curl
curl -s http://localhost:5000/query \
     -H "Content-Type: application/json" \
     -d '{"index": "my_index", "query_type": "count", "query": "natural language processing"}'
```

### 4. (Optional) Raw Python API

`index_dirs` accepts a root directory (auto-discovers sorted subdirs) or an explicit list of shard paths:

```python
from engine.src import InfiniGramMiniEngine

# Option A: single root directory — loads all subdirectories as shards
engine = InfiniGramMiniEngine(
    index_dirs="/path/to/index",
    load_to_ram=False,
    get_metadata=True,
)

# Option B: explicit list of shard directories
engine = InfiniGramMiniEngine(
    index_dirs=["/path/to/index/00", "/path/to/index/01"],
    load_to_ram=False,
    get_metadata=True,
)

engine.count("natural language processing")
# {'count': 83470}

engine.find("natural language processing")
# {'cnt': 83470, 'segment_by_shard': [[442381579355, 442381620985], ...]}

engine.get_doc_by_rank(s=0, rank=442381579355, needle_len=27, max_ctx_len=200)
# {'doc_ix': ..., 'text': '...', 'metadata': {...}}
```

---

## Indexing a new dataset

Supported input formats: `.parquet` (with `text` column), `.jsonl`, `.json.gz`, `.zst`.

```bash
cd infini-gram-mini/indexing
python jobs/index_v2_parquet.py \
    --data_dir /path/to/parquet_files \
    --save_dir /path/to/index_output \
```

The script automatically selects between the few-files and many-files pipeline depending on file count, and produces one index shard per entry in `save_dir/00/`, `save_dir/01/`, etc.

If you hit "too many open files":

```bash
ulimit -n 1048576
# or pass --ulimit 1048576 to the indexing script
```

---

## SLURM example

```bash
#!/bin/bash
#SBATCH --job-name=index
#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=64

cd /path/to/infini-gram-mini/infini-gram-mini/indexing
python jobs/index_v2_parquet.py \
    --data_dir /path/to/data \
    --save_dir /path/to/index \
    --num_shards 10 \
    --mem 500 \
    --cpus $SLURM_CPUS_PER_TASK
```

---

## Citation

If you use infini-gram mini, please cite the original paper:

```bibtex
@misc{xu2025infinigramminiexactngram,
  title={Infini-gram mini: Exact n-gram Search at the Internet Scale with FM-Index},
  author={Hao Xu and Jiacheng Liu and Yejin Choi and Noah A. Smith and Hannaneh Hajishirzi},
  year={2025},
  eprint={2506.12229},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2506.12229},
}
```
