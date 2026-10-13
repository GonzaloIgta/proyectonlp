from datasets import load_dataset

ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
ds.save_to_disk("data/tickets")

# O como CSV:
ds.to_csv("data/tickets.csv", index=False)
