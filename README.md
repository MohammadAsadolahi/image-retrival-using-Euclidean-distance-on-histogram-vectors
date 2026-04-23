<div align="center">

# 🔍 Image Retrieval Using Euclidean Distance on Histogram Vectors

**A content-based image retrieval (CBIR) system that finds visually similar images by comparing color histogram signatures using Euclidean distance.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)](https://matplotlib.org)

</div>

---

## 💡 Core Idea

> Given a query image, retrieve the top-10 most visually similar images from a dataset — without any deep learning, purely through **color distribution comparison**.

Each image is represented as three 256-bin histograms (one per RGB channel). The similarity between two images is measured by summing the **Mean Squared Error (MSE)** across their Red, Green, and Blue histogram vectors:

$$
D(I_1, I_2) = \text{MSE}(H_R^1, H_R^2) + \text{MSE}(H_G^1, H_G^2) + \text{MSE}(H_B^1, H_B^2)
$$

where $H_C^k$ is the 256-bin histogram of channel $C$ for image $k$.

---

## 🏗️ How It Works

```
    Query Image
         │
         ▼
┌──────────────────────┐
│  Split into R, G, B  │
│  channels            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Compute 256-bin     │
│  histogram per       │
│  channel             │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────┐
│  Compare against all images  │
│  using Euclidean distance    │
│  (MSE) on histogram vectors  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────┐
│  Return top-10 most  │
│  similar images      │
└──────────────────────┘
```

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `retrival image on cifar10 dataset using histogram distance.ipynb` | Notebook — retrieval on **CIFAR-10** dataset |
| `retrival image on cifar10 dataset using histogram distance.py` | Standalone script — same CIFAR-10 retrieval |
| `histogrzm_of_a_pic_coreal.ipynb` | Notebook — retrieval on **Corel** dataset (1000 images) |

---

## 📊 Datasets

### CIFAR-10
- **60,000** images (32×32 px) across 10 classes
- Loaded directly via `tensorflow.keras.datasets`
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Corel Dataset
- **1,000** categorized images (100 images × 10 categories)
- Loaded from a local `c/` directory of JPEG files
- Accuracy is evaluated by checking if retrieved images belong to the same 100-image category block

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib tensorflow opencv-python
```

### Run the CIFAR-10 retrieval

```bash
python "retrival image on cifar10 dataset using histogram distance.py"
```

Or open the corresponding `.ipynb` notebook and run all cells.

### Run the Corel retrieval

1. Place your Corel dataset images in a `c/` folder (named `0.jpg` through `999.jpg`)
2. Open `histogrzm_of_a_pic_coreal.ipynb` and run all cells

---

## 📈 Output

For a given query image, the system:

1. **Displays the query image**
2. **Retrieves and displays the top-10 most similar images** ranked by histogram distance
3. **Reports accuracy** — the percentage of retrieved images that belong to the same category as the query

```
++++++++++++++++++ matched images for picture 29.jpg: ++++++++++++++++++
item: 29.jpg  amount of: 70% of similar pictures are from item's category
```

---

## 🔧 Key Functions

| Function | Purpose |
|----------|---------|
| `getHist(channel)` | Computes a 256-bin intensity histogram for a single color channel |
| `vectorDistance(v1, v2)` | Calculates Mean Squared Error between two histogram vectors |
| `imageDistance(img1, img2)` | Computes total distance between two images (sum of R, G, B distances) |
| `plotHistogram(histogram)` | Visualizes a histogram as a bar chart |
| `plotChannel(channel)` | Displays a single color channel as a grayscale image |

---

## 🧠 Future Work

- [ ] Apply a neural network to extract high-level feature vectors and use vector comparison for retrieval
- [ ] Extend to larger and more diverse datasets
- [ ] Benchmark against deep learning-based retrieval methods (e.g., CNN embeddings + cosine similarity)

---

<div align="center">

**Built with NumPy, Matplotlib, and the elegance of simple math.**

</div>
