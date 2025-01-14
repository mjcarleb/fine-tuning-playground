from datasets import load_dataset

dataset = load_dataset("lamini/lamini_docs")
print("Available keys:", dataset["train"].features)
print("\nFirst example:", dataset["train"][0])