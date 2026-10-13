import os

from datasets import load_dataset, load_from_disk

PATH = "data/tickets"

if not os.path.exists(os.path.join(PATH, "dataset_dict.json")):
    load_dataset("Tobi-Bueck/customer-support-tickets").save_to_disk(PATH)

# Carga unicamente el train desde disco
train = load_from_disk(os.path.join(PATH, "train"))

print(train)
