from datasets import load_dataset
from tqdm.auto import tqdm

dataset = load_dataset("/home/christoph.minixhofer/libritts-r-aligned/libritts-r-aligned.py")

train = dataset["train"]
# shuffle train
train = train.shuffle()
dev = dataset["dev"]

dev_spk = set(dev["speaker"])

train_spk = set(train["speaker"])

print(f"dev has {len(dev_spk)} speakers")
print(f"train has {len(train_spk)} speakers")

# difference
print(f"train - dev: {train_spk - dev_spk}")
print(f"dev - train: {dev_spk - train_spk}")