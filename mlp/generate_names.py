# Generate up to 100 random names in console based on mlp language model
import random

import torch
import torch.nn.functional as F

START_TOKEN = "("
END_TOKEN = ")"
NUM_NAMES = 100  # How many names to generate
CONTEXT_SIZE = 3  # How many chars as context when predicting next one
TRAIN_SIZE = 0.9  # The remaining is kept for validation
EMBEDDING_DIM = 10  # Dimension of each char embedding
LAYER_SIZE = 60  # Number of units in layer 1
NUM_ITER = 100_000  # Number of iterations when training
BATCH_SIZE = 4  # Minibatch size in each iteration

# Load dataset
with open("data/names.txt", "r") as f:
    names = f.read().splitlines()
print(f"{len(names)} names loaded. Examples: {names[:3]}..")

# Create vocab with mapping utils
vocab = [START_TOKEN] + sorted(list(set("".join(names)))) + [END_TOKEN]
stoi = {s: i for i, s in enumerate(vocab)}
itos = {i: s for i, s in enumerate(vocab)}
print(f"Vocab of size {len(vocab)} created")


def init_dataset(names, context_size=3):
    X, Y = [], []
    for name in names:
        context = [stoi[START_TOKEN]] * context_size
        for char in name + END_TOKEN:
            idx = stoi[char]
            X.append(context)
            Y.append(idx)
            context = context[1:] + [idx]  # Context for next char in the name
    X, Y = torch.tensor(X), torch.tensor(Y)
    return X, Y


# Setup dataset
print(f"Setup {TRAIN_SIZE*100}% for training and {100-TRAIN_SIZE*100}% for validation")
random.shuffle(names)
cutoff = int(TRAIN_SIZE * len(names))
X_train, y_train = init_dataset(names[:cutoff], context_size=CONTEXT_SIZE)
X_val, y_val = init_dataset(names[cutoff:], context_size=CONTEXT_SIZE)
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

# Initiate parameters for the model
vocab_size = len(vocab)

C = torch.randn((vocab_size, EMBEDDING_DIM))  # Embeddings
W1 = torch.randn((CONTEXT_SIZE * EMBEDDING_DIM, LAYER_SIZE))
b1 = torch.randn(LAYER_SIZE)
W2 = torch.randn((W1.shape[1], vocab_size))
b2 = torch.randn(vocab_size)
parameters = [C, W1, b1, W2, b2]


# Train network
for p in parameters:
    p.requires_grad = True

train_loss = []
for i in range(NUM_ITER):
    batch_idxs = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))

    embeddings = C[X_train[batch_idxs]]
    h = torch.tanh(embeddings.view(-1, W1.shape[0]) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y_train[batch_idxs])

    # Backward
    for p in parameters:  # Reset gradients before backward pass
        p.grad = None
    loss.backward()

    # Update gradients with sharper learning rate in the start
    lr = 0.1 if i < 50_000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    train_loss.append(loss.item())

print(f"Finished training. Last 10 minibatch losses:\n{train_loss[-10:]}")

# Compare train and valid loss
embeddings = C[X_train]
h = torch.tanh(embeddings.view(-1, W1.shape[0]) @ W1 + b1)
logits = h @ W2 + b2
training_loss = F.cross_entropy(logits, y_train)

embeddings = C[X_val]
h = torch.tanh(embeddings.view(-1, W1.shape[0]) @ W1 + b1)
logits = h @ W2 + b2
valid_loss = F.cross_entropy(logits, y_val)

print(f"X_train loss: {training_loss.item()}")
print(f"X_val loss: {valid_loss.item()}")

# Generate names
print(f"\n{'#'*50}\nGenerating {NUM_NAMES} names\n{'#'*50}\n")
for _ in range(NUM_NAMES):
    name = []
    context = [stoi[START_TOKEN]] * CONTEXT_SIZE
    while True:
        embeddings = C[torch.tensor([context])]  # (1, context_size, embedding_dim)
        h = torch.tanh(embeddings.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        idx = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [idx]
        if idx == stoi[END_TOKEN]:
            break
        name.append(itos[idx])
    name = "".join(name)
    if name not in names:
        print(f"Hello there Mr. {name}")
