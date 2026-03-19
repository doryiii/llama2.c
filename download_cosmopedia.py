from datasets import load_dataset
import json
import os
import shutil
from tqdm import tqdm

DATA_CACHE_DIR = "data"
out_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")

# We will move existing TinyStories out of the way instead of deleting it
backup_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data_backup")
if os.path.exists(out_dir) and not os.path.exists(backup_dir):
    print(f"Backing up existing {out_dir} to {backup_dir}...")
    shutil.move(out_dir, backup_dir)
elif os.path.exists(out_dir):
    shutil.rmtree(out_dir)

os.makedirs(out_dir, exist_ok=True)

shard_size = 100000
shard_idx = 0
examples = []

def process_split(config_name, max_shards):
    global shard_idx, examples
    print(f"Loading Cosmopedia {config_name}...")
    ds = load_dataset("HuggingFaceTB/cosmopedia", config_name, split="train", streaming=True)
    
    shards_written = 0
    for row in tqdm(ds):
        examples.append({"story": row["text"]})
        if len(examples) == shard_size:
            with open(os.path.join(out_dir, f"cosmo{shard_idx:02d}.json"), "w") as f:
                json.dump(examples, f)
            examples = []
            shard_idx += 1
            shards_written += 1
            if shards_written >= max_shards:
                break

# Download 15 shards of stories (1.5M stories, avg ~800 tokens = ~1.2B tokens)
process_split("stories", 15)

# Download 5 shards of web_samples_v2 (500k samples, avg ~1000 tokens = ~500M tokens)
process_split("web_samples_v2", 5)

if examples:
    with open(os.path.join(out_dir, f"cosmo{shard_idx:02d}.json"), "w") as f:
        json.dump(examples, f)
        
print(f"Done downloading Cosmopedia subset. Total shards: {shard_idx + 1}")
