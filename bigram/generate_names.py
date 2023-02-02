# Generate 100 random names in console based on bigram language model
import torch

START_TOKEN = "("
END_TOKEN = ")"
NUM_NAMES = 100

# Load dataset
with open("data/names.txt", "r") as f:
    names = f.read().splitlines()
print(f"{len(names)} names loaded. Examples: {names[:3]}..")

# Create vocab with mapping utils
vocab = [START_TOKEN] + sorted(list(set("".join(names)))) + [END_TOKEN]
stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for i, s in enumerate(vocab)}
print(f"Vocab of size {len(vocab)} created")

# Create Bigram Matrix
B = torch.zeros((len(vocab), len(vocab)), dtype=torch.int32)
for name in names:
    name_chars = [START_TOKEN] + list(name) + [END_TOKEN]
    for c1, c2 in zip(name_chars, name_chars[1:]):
        B[stoi[c1], stoi[c2]] += 1
print(f"Bigram matrix of shape {B.shape} created")

# Create Probability Matrix
P = (B + 1).float()
P /= P.sum(1, keepdims=True)
print(f"Probability matrix of shape {P.shape} created")

# Generate names
print(f"\n{'#'*50}\nGenerating {NUM_NAMES} names\n{'#'*50}\n")
for i in range(NUM_NAMES):
    name = []
    c1 = stoi[START_TOKEN]
    while True:
        c2_probs = P[c1]
        c2 = torch.multinomial(c2_probs, num_samples=1, replacement=True).item()
        if c2 == stoi[END_TOKEN]:
            break
        name.append(itos[c2])
        c1 = c2
    print(f"Hello there Mr. {''.join(name)}")
