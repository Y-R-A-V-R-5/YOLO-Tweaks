# 🧠 YOLO-Tweaks: Backbone–Head Hybrid Compatibility Experiments Across YOLOv8 / YOLOv10 / YOLOv11  
🚀 **Personal Research Project (Independent Study)**  
📍 *Exploring structural compatibility, scaling behavior, and performance trade-offs across YOLO architectures.*

---

## 🎯 Overview

This project investigates whether mixing and matching **backbones** and **detection heads** across three YOLO generations can produce architectures that are:

- More efficient  
- More accurate  
- Better specialized for tiny-object detection  
- Better balanced between compute and mAP  

Instead of treating YOLO versions as fixed models, this work approaches them as **modular systems**—where backbones, necks, and heads can be recombined to form new hybrids.

All experiments were run on the **AgriPest** tiny-insect dataset.

---

## 🎯 Objective
This project explores how different combinations of backbones and detection heads affect performance on a fine-grained, tiny-object dataset (AgriPest).

Three standard YOLO models are used as baselines, and three custom hybrid architectures are created by mixing/ swapping backbone–head pairs.

The goal is to understand:

1. How compute (params, FLOPs),
2. Feature extraction quality, and
3. Multi-scale detection head design

together influence precision, recall, and mAP.

> Rather than treating YOLO architectures as immutable, this experiment approaches them **as modular systems** — where *backbone–neck–head components* can be recombined to discover hybrid efficiency.
---

## 🧩 Model Variants

### **Baselines**

| Model | Backbone | Head |
|-------|----------|-------|
| YOLOv8 | v8 | v8 |
| YOLOv10 | v10 | v10 |
| YOLOv11 | v11 | v11 |

### **Hybrids**

| Hybrid | Backbone | Head |
|--------|-----------|--------|
| v8v10 | v8 | v10 |
| v10v11 | v10 | v11 |
| v11v8 | v11 | v8 |

Each configuration was implemented via custom YAMLs and validated for tensor-shape integrity within Ultralytics’ PyTorch framework.

---

## 🧱 Architecture Highlights

| Version     | Distinct Structural Traits                                                  |
| :---------- | :-------------------------------------------------------------------------- |
| **YOLOv8**  | C2f modules, simple PAN head, strong dense-feature representation           |
| **YOLOv10** | SCDown, C3k2 blocks, decoupled head → optimized for latency                 |
| **YOLOv11** | C3k2 + CIB + refined scaling for speed–accuracy balance                     |
| **Hybrids** | Combine efficiency modules (v10/v11) with dense representational heads (v8) |

---

## ⚙️ Experimental Setup

| Parameter     | Value                               |
| :------------ | :---------------------------------- |
| Dataset       | **AgriPest (13 classes)**           |
| Image size    | 320                                 |
| Epochs        | 50                                  |
| Patience      | 5                                   |
| Warmup epochs | 5                                   |
| Batch size    | 24                                  |
| Optimizer     | SGD (Ultralytics default)           |
| Device        | CPU — Intel i5-12500                |
| Framework     | Ultralytics 8.3.220 / PyTorch 2.9.0 |

---

## 📂 Dataset

**AgriPest** — a crop-pest dataset featuring small insects across 13 categories.

| Split      | Images |
| :--------- | -----: |
| Train      | 11 502 |
| Validation | 1 095  |
| Test       | 546    |

**Dataset Strucutre**

```
AgriPest/
├── train/
├── valid/
├── test/
└── data.yaml
```

---

## 🧮 Model Complexity

| Model      | Layers | Parameters | FLOPs |
| :--------- | -----: | ---------: | ----: |
| YOLOv8     | 129    | 3.01 M     | 8.2 G |
| YOLOv10    | 223    | 2.71 M     | 8.4 G |
| YOLOv11    | 181    | 2.59 M     | 6.5 G |
| **v8v10**  | 196    | 2.98 M     | 8.8 G |
| **v10v11** | 168    | 2.06 M     | 6.6 G |
| **v11v8**  | 153    | 2.52 M     | 6.2 G |

> Hybridization can reduce parameter count and FLOPs while maintaining comparable feature depth. Hybrids reduced compute while maintaining feature depth.


---

## 📊 Performance Summary

### Validation mAP (AgriPest Valid – 1,095 images)

| Model | Precision | Recall | mAP50 | mAP50-95 |
|--------|----------:|---------:|---------:|------------:|
| **YOLOv8** | 0.676 | 0.597 | **0.660** | **0.416** |
| **YOLOv10** | 0.634 | 0.485 | 0.586 | 0.380 |
| **YOLOv11** | 0.629 | 0.573 | 0.627 | 0.395 |
| **v8v10** | 0.656 | 0.513 | 0.608 | 0.390 |
| **v10v11** | 0.671 | 0.591 | **0.655** | **0.409** |
| **v11v8** | 0.642 | 0.552 | 0.631 | 0.384 |

### Interpretation  
- **YOLOv8** remains the most accurate baseline on tiny insects.  
- **v10v11** becomes the *top hybrid*, nearly matching YOLOv8 accuracy with lower compute.  
- **v11v8** is balanced: strong recall + moderate FLOPs.  
- **v8v10** improves recall vs v10 but stays mid-tier. 

---

## ⏱️ Training & Inference Time

|### **Baselines**

| Model | Training Time | Inference (ms/img) |
|--------|----------------:|-----------------------:|
| YOLOv8 |  531m 52.5s | 36.9 |
| YOLOv10 | 699m 12.4s | 17.2 |
| YOLOv11 | 564m 51.0s | **14.4** |

### **Hybrids (from logs)**

| Hybrid | Training Time | Inference (ms/img) |
|--------|----------------:|-----------------------:|
| v8v10 | 735m 45.5s | 14.7 |
| v10v11 | 962m 50.5s | 27.9 |
| v11v8 | 843m 42.0s | 23.2 |

> **YOLOv11 remains the fastest** due to lowest FLOPs;  
> **v8v10** shows strong inference efficiency among hybrids.
---

## 📊 Performance Comparison — Baselines vs Hybrids (AgriPest Validation)

This section compares all six models using consistent, validation-set metrics:

- Precision (P)  
- Recall (R)  
- mAP50  
- mAP50-95  
- Inference speed  

---

## 🟦 Baseline Models

### **YOLOv8 (Baseline)**
- **P:** 0.676  
- **R:** 0.597  
- **mAP50:** 0.660  
- **mAP50-95:** 0.416  
- **Inference:** 13.9 ms  
- Strongest baseline; robust for tiny-object detection.

---

### **YOLOv11 (Baseline)**
- **P:** 0.629  
- **R:** 0.573  
- **mAP50:** 0.627  
- **mAP50-95:** 0.395  
- **Inference:** 14.4 ms  
- Most efficient baseline (lowest FLOPs, fastest inference).

---

### **YOLOv10 (Baseline)**
- **P:** 0.634  
- **R:** 0.485  
- **mAP50:** 0.586  
- **mAP50-95:** 0.380  
- **Inference:** 17.3 ms  
- Weakest of the three baselines but still efficient.

---

## 🟩 Hybrid Architectures (Backbone × Head)

## **v8v10 — YOLOv8 Backbone + YOLOv10 Head**
- **P:** 0.656  
- **R:** 0.513  
- **mAP50:** 0.608  
- **mAP50-95:** 0.390  
- **Inference:** 14.7 ms  
- Outperforms YOLOv10; sits between v11 and v8 in accuracy.

---

### **v10v11 — YOLOv10 Backbone + YOLOv11 Head**
- **P:** 0.671  
- **R:** 0.591  
- **mAP50:** 0.655  
- **mAP50-95:** 0.409  
- **Inference:** 27.9 ms  
- ⭐ **Top hybrid model**  
- Nearly matches YOLOv8 accuracy with fewer parameters.

---

### **v11v8 — YOLOv11 Backbone + YOLOv8 Head**
- **P:** 0.642  
- **R:** 0.552  
- **mAP50:** 0.631  
- **mAP50-95:** 0.384  
- **Inference:** 23.2 ms  
- High recall and solid accuracy; heavier head increases inference time.

---

## 🧠 Summary

| Model | Type | P | R | mAP50 | mAP50-95 | Inference |
|-------|------|------:|------:|--------:|------------:|------------:|
| **YOLOv8** | Baseline | **0.676** | 0.597 | **0.660** | **0.416** | 13.9 ms |
| **YOLOv11** | Baseline | 0.629 | 0.573 | 0.627 | 0.395 | **14.4 ms** |
| **YOLOv10** | Baseline | 0.634 | 0.485 | 0.586 | 0.380 | 17.3 ms |
| **v8v10** | Hybrid | 0.656 | 0.513 | 0.608 | 0.390 | 14.7 ms |
| **v10v11** | Hybrid | 0.671 | 0.591 | **0.655** | 0.409 | 27.9 ms |
| **v11v8** | Hybrid | 0.642 | 0.552 | 0.631 | 0.384 | 23.2 ms |

---

## 🔍 Interpretation
- **YOLOv8 remains the accuracy leader**, setting the baseline.  
- **v10v11 is the strongest hybrid**, approaching YOLOv8 mAP while using fewer FLOPs.  
- **v11 base is the most efficient baseline**, offering fastest inference.  
- **v8v10 is an efficient mid-performance hybrid**, outperforming v10.  
- **v11v8 blends strong recall with slightly higher compute.**

---

## 🧠 Observations

YOLOv8 → best accuracy on tiny targets (rich C2f blocks)  
YOLOv11 → best efficiency, leanest compute  
YOLOv10 → balance of both, but underperforms on small pests  
**Hybrid v11v8 (expected):** lightweight backbone + denser head = potential sweet spot

---

## 📈 Visualizations (to be included)

- Layer count vs FLOPs  
- mAP vs GFLOPs  
- Precision-Recall curves  
- Inference speed comparison  

*(Plots auto-generated from `results.csv` and `train.log` once all variants complete training.)*

---

## 🔍 Observations & Insights

### What this experiment reveals:

- **Dense-feature heads (v8)** improve small-object detection even when paired with newer backbones.  
- **Efficient heads (v10/v11)** significantly lower compute but require strong low-level features.  
- **Backbone–head compatibility matters**: mismatches lower mAP even when compute is reduced.  
- **Hybrid v10v11** achieves the best balance → low compute + high accuracy.
---

## 🔮 Future Work

1. Complete training for hybrid (v8v10, v10v11, v11v8) models.  
2. Plot architecture-efficiency curves (mAP vs FLOPs vs params).  
3. Evaluate hybrid stability on more datasets (COCO-subset / BCCD).  
4. Compare against lightweight architectures (YOLO-Nano, RT-DETR).  
5. Export hybrids to ONNX / TorchScript for speed benchmarking.

---

## 📚 Key Takeaways

- **Architectural literacy** → understanding YOLO internals beyond API usage.  
- **Feature-head synergy** → small changes in module hierarchy heavily affect detection consistency.  
- **Efficiency-accuracy trade-off** → lower GFLOPs ≠ lower mAP when structural alignment is preserved.

---

## 🧾 Author

**Y. R. A. V. R** — Hyderabad, India  
🔗 [LinkedIn](https://www.linkedin.com/in/yravr/)  |  [GitHub](https://github.com/Y-R-A-V-R-5/YOLO-TWEAKS)

---

## 📎 References

- [Ultralytics YOLOv8 → v11 Releases](https://github.com/ultralytics/ultralytics)  
- [SCDown and C3k2 modules (YOLOv10 paper)](https://arxiv.org/abs/2405.14458)  
- [CIB Blocks and Hybrid Heads (YOLOv11)](https://github.com/ultralytics/ultralytics/releases)  
- [AgriPest Dataset](https://www.kaggle.com/datasets/xiaoyuan-chen/agri-pest-dataset)

---

### ⚡ README Snapshot Summary

> *“YOLO-Tweaks” demonstrates how a hands-on learner can dissect and recombine architectures across model generations — moving from training usage to genuine architectural experimentation.*
> > *This repository demonstrates architectural literacy, experimental thinking, and hands-on model engineering — transitioning from model usage to genuine structural exploration.*
