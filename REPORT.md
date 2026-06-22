Here is the entire report optimized specifically for GitHub Markdown compatibility. You can click the **"Copy"** button on the code block below and paste it directly into your `docs/engineering_report.md` or `REPORT.md` file in VS Code!

```markdown
# Engineering Project Report: End-to-End EMNIST Character Classifier
**System Architecture, Deep Learning Pipeline, and MLOps Deployment**

---

## 1. Executive Summary
This report outlines the design, implementation, and deployment of an end-to-end machine learning system capable of performing real-time handwritten character recognition (covering both numerical digits and alphabetical letters). The system comprises an optimized deep learning model trained on the EMNIST Balanced dataset using PyTorch, a high-performance asynchronous API layer built with FastAPI, and a containerized infrastructure deployed on Hugging Face Spaces using Docker and an optimized Python runtime environment. 

---

## 2. Dataset Analysis & Problem Formulation
The target task requires classifying complex handwritten structures across a high-dimensional label space consisting of alphanumeric characters. 

* **Dataset Selection:** The model is trained on the **EMNIST Balanced** dataset, which consists of 47 balanced classes with a total test pool of 18,800 samples. 
* **Label Space Challenges:** Unlike standard MNIST (10 digits), EMNIST Balanced scales the output complexity to 47 classes. This expansion introduces significant inter-class ambiguity due to structural similarities in human handwriting (e.g., distinguishing uppercase `O` from digit `0`, uppercase `I` from digit `1`, lowercase `l` from uppercase `L`, and lowercase `g` from lowercase `q`).

![Sample data](assets/samples.png)

---

## 3. Machine Learning Pipeline Architecture

### 3.1 Data Engineering & Input Pipeline
To prevent memory exhaustion during training and maximize computational throughput, a decoupled three-stage asynchronous data loading pipeline was engineered:
1. **Transforms:** Raw structural matrices are ingested, converted into floating-point tensors, and scaled to a standardized pixel value distribution through normalization.
2. **Dataset:** Implemented using PyTorch’s lazy-loading abstractions (`Dataset` class), ensuring that individual sample files or memory arrays are indexed and fetched dynamically from disk on demand, keeping the host system's RAM consumption strictly bounded.
3. **DataLoader:** Combines individual data samples into mini-batches, handles data shuffling across epochs to disrupt sequential ordering biases, and handles memory pinning for parallel host-to-device transfers.

### 3.2 Regularization via Data Augmentation
To counter real-world variations introduced by an open-canvas web UI (such as variable drawing scales, off-center brush strokes, and slanted writing styles), geometrical regularization was introduced directly into the training pipeline via a `RandomAffine` transformation layer.
* **Configuration:** Applied random rotations within a $\pm15^\circ$ boundary alongside multi-axis translation and scaling constraints.
* **System Impact:** This runtime modification intentionally made training samples harder than the pristine, centered validation set. This structural regularization technique prevented the model from memorizing pixel-coordinate mappings, forcing it to learn invariant stroke configurations.

### 3.3 Neural Network Topology (Model Architecture)
The network is designed as a Multi-Layer Perceptron (MLP) using PyTorch’s object-oriented `nn.Module` infrastructure. It optimizes $\sim240,000$ trainable parameters across its dense connections.

| Layer Type | Component | Structural Dimensionality / Operational Parameters | Act. Function |
| :--- | :--- | :--- | :--- |
| **Input** | Flatten Layer | Ingests $28 \times 28$ grayscale pixels $\rightarrow$ vectorizes to $784$ structural nodes | N/A |
| **Hidden 1** | Fully Connected (`FC1`) | Maps $784$ input connections down to $256$ intermediate features | ReLU |
| **Hidden 2** | Fully Connected (`FC2`) | Maps $256$ hidden features down to $128$ compressed representations | ReLU |
| **Output** | Fully Connected (`FC3`) | Maps $128$ bottleneck dimensions directly to the $47$ unnormalized target logit classes | Softmax (Implicit via Loss) |

### 3.4 Training Dynamics & Compute Scheduling
* **Objective Function:** Optimization was governed by **Cross-Entropy Loss**, measuring divergence between predicted logit distributions and one-hot encoded ground truth vectors.
* **Optimization Engine:** The **Adam Optimizer** was selected with a base learning rate of $\eta = 0.001$. Adam's adaptive tracking of first and second-moment vector gradients applies per-parameter learning adjustments, ensuring rapid momentum during initial flat loss landscapes and micro-scale damping near sharp local minima.
* **Hardware-Agnostic Accelerator Scheduling:** Execution blocks dynamically adjust to running hardware resources. The environment automatically tests for CUDA-capable graphics processors:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```

The network graph is mapped to the device once upon instantiation. During active training iterations, input vectors and target matrices are asynchronously streamed onto the target device buffer inside the batch loop, minimizing bottleneck latency across the PCIe bus.

---

## 4. Experimental Results & Performance Evaluation

### 4.1 Evaluation Metrics

The training sequence yielded highly stable learning progression across targeted epochs:

* **Final Generalization Metrics:** The architecture demonstrated an overall **Test Accuracy of 83.93%** when evaluated against the clean EMNIST Balanced test distribution (18,800 sample validation pool).
* **Divergence Anomalies:** A distinct behavioral characteristic emerged where the overall training accuracy hovered around **~80%**, trailing behind the **~84%** test accuracy. This negative divergence is a direct consequence of the aggressive `RandomAffine` data augmentation strategy. Because training data underwent continuous geometric distortion, the model confronted a systematically more difficult distribution during training than it did during pristine validation evaluations, verifying excellent structural generalization.

![Training Graph](assets/training_curves.png)

### 4.2 Error Analysis and Confusion Matrix Inspection

Analysis of the generated confusion matrix reveals isolated points of degradation:

1. **Numerical Dominance:** Discrete numerical classes ($0\text{--}9$) maintained the highest accuracy rates due to their distinctive typographical structures.
2. **Structural Ambiguity Blocks:** The heaviest classification overlaps occurred exclusively inside multi-class cross-character groupings sharing nearly identical stroke features, notably `I ↔ 1`, `O ↔ 0`, `l ↔ L`, and `g ↔ q`.
3. **Case Sensitivity Friction:** Lowercase alphabetical variations represented the hardest overall classification target, as their variations frequently overlap with corresponding uppercase shapes when drawn on an open canvas grid.

![Confusion Matrix](assets/confusion_matrix.png)

---

## 5. Production Engineering & MLOps Infrastructure

```text
emnist-classifier/
├── app.py                  # FastAPI server hosting inference handlers
├── requirements.txt        # CPU-optimized dependency specifications
├── Dockerfile              # Layer-cached container instructions
├── README.md               # Hugging Face Spaces configuration schema
├── templates/
│   └── index.html          # HTML5 interactive canvas UI
└── model/
    ├── model.py            # MLP graph architecture class
    └── emnist_model.pth    # Serialized model tensor weights (~900 KB)

```

### 5.1 Application Layer & Runtime Execution

The application utilizes a decoupled web architecture separating frontend interaction from backend deep learning inference.

* **The Web Client Interface:** The application interface serves an interactive HTML5 canvas backed by vanilla JavaScript. Users sketch characters directly into the browser, which captures bounding coordinates, normalizes drawing weights, and streams the spatial data down to the host server API via asynchronous payload requests.

![Demo](assets/demo.gif)

* **The Asynchronous API Server:** Implemented via **FastAPI**, which handles non-blocking request routing. The root path serves the web interface from the local workspace directory using optimized system paths:
```python
@app.get("/")
async def read_index():
    return FileResponse(os.path.join("templates", "index.html"))

```


Upon receiving base64 image data payloads at the prediction route, the backend interceptor converts the input, reshapes the dimensions into an isolated $(1, 1, 28, 28)$ tensor shape, applies normalization matches, passes it through the pre-loaded network graph, and returns the classification mappings as a structured JSON object.

### 5.2 Containerization & Cloud Image Size Optimization

To achieve robust deployment containment, the application was packaged into a customized container environment using **Docker**.

* **The Problem:** Standard default deployment templates containing deep learning frameworks pull in heavy dependencies (e.g., full CUDA toolkit binaries, development headers), blowing up base images past 2.0 Gigabytes. Such configurations increase build durations, degrade scaling performance, and introduce massive attack surfaces.
* **The Solution:** By constructing a clean `Dockerfile` from scratch and strictly targeting CPU-only architecture distribution packages inside the runtime requirements file (`--index-url https://download.pytorch.org/whl/cpu`), all heavy GPU compilation libraries were excluded. This slashed the ultimate runtime footprint down to **~300 Megabytes**, accelerating startup container spinning and reducing cold boot delays.
* **Security Constraints:** The execution image is hardened to operate under restrictive non-root user permissions, strictly conforming to modern enterprise access control practices required by secure cloud providers.

### 5.3 Git Integration & Dual-Remote Version Control Topology

A unified configuration pipeline was implemented to allow continuous maintenance of both the application host server and the public open-source portfolio code without duplicating terminal work.

```text
                  ┌───────────────┐
                  │ Local Project │
                  │  Repository   │
                  └───────┬───────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    git push origin main        git push hf main
 ┌─────────────────────┐     ┌─────────────────────┐
 │ GitHub Repository   │     │ Hugging Face Spaces │
 │ (Public Portfolio)  │     │ (Live App Engine)   │
 └─────────────────────┘     └─────────────────────┘

```

* **The Binary Storage Block:** Production models rely on serialized binary configuration matrices (e.g., the trained model checkpoint `emnist_model.pth` and portfolio performance assets like `demo.gif`). Standard version control architectures block these binaries upon push operations due to size limitations and differential line-tracking errors.
* **Git LFS Architecture:** The system was configured to leverage **Git LFS (Large File Storage)**, decoupling the physical binary assets from standard tracking trees. By tracking targets through explicit `.gitattributes` constraints (`model/emnist_model.pth` and `assets/*`), binary streams are routed securely through dedicated asset servers, while Git passes lightweight text pointers across the main repository tracking line.
* **Dual-Remote Routing Pipeline:** The local repository was configured with isolated terminal routing paths. Code changes can be selectively or concurrently pushed across independent operational destinations:
* `origin` links directly to a public GitHub repository for documentation, portfolio review, and code-base distribution.
* `hf` points directly to the active Hugging Face Spaces platform, where a tracking post-commit hook immediately reads configuration parameters declared in the `README.md` metadata frontmatter block, builds the revised container layers, and updates the application live on the web.



---

## 6. Conclusion

The architecture detailed in this report demonstrates a successful translation of deep learning design into an optimized cloud application. By solving constraints across input pipelines, network regularizations, data-structure formats, container sizing, and multi-endpoint tracking paths, the resulting application achieves fast response times, high deployment portability, and scalable maintenance.

```

```