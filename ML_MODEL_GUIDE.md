# ML Model Guide (Beginner-Friendly)

This repo contains **two small training examples** in the notebook:
- `notebooks/gpu-ops-test.ipynb` — *Training Example 1*: synthetic (learnable) classification
- `notebooks/gpu-ops-test.ipynb` — *Training Example 2*: MNIST digits + mixed precision

They use the same core idea: **train a neural network to predict a class label**.

---

## What problem are we solving?

Both examples are **multi-class classification** problems.

- **Input**: a vector (a list of numbers) or an image
- **Output**: one of *N* categories (classes)

Examples of multi-class classification in real life:
- Email routing: {billing, support, sales, spam}
- Image labeling: {cat, dog, horse, ...}
- Document topic: {sports, politics, tech, ...}

In this repo, we use:
- **10 classes** in both examples

---

## What model is used?

Both examples use a **feed-forward neural network** (also called a **dense neural network**, **fully-connected network**, or **MLP**).

### Intuition (simple explanation)

A dense neural network is like a flexible function that learns a mapping:

- “When these input numbers look like *this*, output a high score for class 7”
- “When they look like *that*, output a high score for class 2”

It learns by adjusting internal parameters called **weights** (numbers inside the model).

### Architecture used here

**Example 1 (synthetic data):**
- Input: 100 features
- Hidden layer: 64 units with ReLU
- Output layer: 10 units with softmax

**Example 2 (MNIST):**
- Input: 28×28 image (flattened into a vector)
- Hidden layer: 128 units with ReLU
- Hidden layer: 64 units with ReLU
- Output layer: 10 units with softmax

---

## What does the model output?

The last layer uses **softmax**, which converts raw scores into **probabilities**.

For 10 classes, softmax outputs a 10-number vector like:

- `[0.01, 0.02, 0.00, 0.90, 0.01, 0.01, 0.01, 0.02, 0.01, 0.01]`

Interpretation:
- The model believes class **3** is most likely (~90%).

---

## What is the goal during training?

Training tries to make the model’s predicted probabilities match the true label.

### Loss (what the optimizer tries to minimize)

Both examples use:
- **categorical cross-entropy loss**

Simple way to think about it:
- If the model is confident and correct → loss is small
- If the model is confident and wrong → loss is large

### Optimizer (how the model learns)

Both examples use:
- **Adam** optimizer

Simple way to think about it:
- Adam decides how to adjust weights to reduce loss.

### Metrics (human-friendly score)

Both examples report:
- **accuracy** (percentage of correct predictions)

Accuracy is easier to interpret than loss, but loss is what training actually optimizes.

---

## Example 1: Synthetic (learnable) classification

### Why include a synthetic example?

This example is designed for learning.

It avoids common beginner pain points:
- No dataset download
- No file formats
- No image preprocessing

Instead, it focuses on *the training loop and the concepts*.

### What is the dataset?

We generate input features `X` as random numbers:
- `X` is shape `(num_samples, num_features)`

Then we generate labels using a hidden rule:

1. Create a hidden weight matrix `W`
2. Compute scores (logits):
   - `logits = X @ W + noise`
3. The label is the index of the largest score:
   - `y = argmax(logits)`

Why this matters:
- The labels are **not random**.
- There is a real pattern: inputs and labels are connected.
- A neural network can learn this pattern, so accuracy should improve.

### Practical meaning

This is like a simplified “tabular data” classification task:
- features might be: age, income, number of purchases, etc.
- label might be: {low risk, medium risk, high risk}

Synthetic data is not used to deploy a real model, but it’s extremely useful to:
- test code paths
- test GPU/CPU behavior
- understand training dynamics

---

## Example 2: MNIST digits + mixed precision

### Why include MNIST?

MNIST is a classic beginner dataset:
- Input: images of handwritten digits (0–9)
- Output: the digit class

It makes results easier to interpret:
- If accuracy is high, the model is actually doing something meaningful.

### What does mixed precision mean?

**Mixed precision** means we use **float16** for many computations, which can be faster on modern NVIDIA GPUs (Tensor Cores), while keeping some critical values in **float32** for stability.

Why it can help:
- Faster math on GPU
- Lower memory bandwidth usage

Why we force the output layer dtype to float32:
- Softmax + loss can be numerically sensitive
- float32 helps avoid overflow/underflow issues

### Practical meaning

This is closer to real workflows:
- Real dataset
- Preprocessing (normalizing pixels)
- A simple model you can later replace with CNNs or more advanced architectures

---

## Key ML terms used in the notebook

### Features

“Features” are the input numbers.
- Example 1: 100 random features per sample
- Example 2: pixel values from an image

### Labels

“Labels” are the correct answers.
- For 10 classes, labels can be represented as:
  - **integer**: `7`
  - **one-hot vector**: `[0,0,0,0,0,0,0,1,0,0]`

### Epoch

An **epoch** = one full pass through the training dataset.

### Batch size

Instead of updating the model after every single sample, we update after a batch (a chunk).

- Larger batch:
  - often faster per epoch
  - uses more memory
- Smaller batch:
  - slower
  - uses less memory

### Train / Validation / Test

- **Train**: data the model learns from
- **Validation**: data used during training to detect overfitting
- **Test**: final “report card” on unseen data

---

## How to use these models in a practical manner

The models in this repo are intentionally simple, but the workflow is the same as real projects:

1. **Define the prediction task**
   - What are inputs? What are outputs?
2. **Prepare data**
   - cleaning, normalization, label encoding
3. **Choose a model**
   - start simple, then add complexity
4. **Train**
   - monitor loss/accuracy
5. **Evaluate**
   - test set performance
6. **Deploy/use**
   - run `model.predict(...)` on new data

Where this exact type of model is used:
- Tabular classification (fraud detection, churn prediction, risk scoring)
- Simple image classification baselines
- Text classification after converting text to numeric features (embeddings)

---

## Suggested beginner experiments (safe changes)

Try these one at a time:

### Example 1 (synthetic)
- Increase epochs from 5 → 20 and see if accuracy improves
- Change hidden units from 64 → 128
- Reduce `noise_scale` (easier problem) or increase it (harder)

### Example 2 (MNIST)
- Increase epochs from 5 → 10
- Change batch size 64 → 128 (watch speed and memory)
- Add a dropout layer to reduce overfitting

---

## GPU note (why this repo cares)

GPU acceleration helps because training involves lots of matrix multiplications.

This repo’s notebook includes checks and setup steps so:
- it uses GPU when available
- it falls back to CPU when not

On Linux/WSL, TensorFlow may sometimes need CUDA Toolkit tools like `ptxas` for some GPU compilation paths. The notebook prints whether `ptxas`/`nvcc` are found.
