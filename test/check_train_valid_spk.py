from datasets import load_dataset
from tqdm.auto import tqdm

dataset = load_dataset("/home/christoph.minixhofer/libritts-r-aligned/libritts-r-aligned.py")

train = dataset["train"]
# shuffle train
train = train.shuffle()
dev = dataset["dev"]

dev_spk = set(dev["speaker"])

rmv_spk = []

for item in tqdm(dataset["train"]):
    spk = item["speaker"]
    if spk in dev_spk:
        # remove speaker from dev_spk
        dev_spk.remove(spk)
        rmv_spk.append(spk)
        # print dev_spk length
        print(len(dev_spk))
    elif spk not in rmv_spk:
        raise ValueError(f"Speaker {spk} not in dev set!")
    

if len(dev_spk) > 0:
    print("dev_spk not empty!")
    print(dev_spk)