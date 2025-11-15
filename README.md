# 🧠 YOLO-Tweaks: Backbone–Head Compatibility Experiments Across YOLOv8 / YOLOv10 / YOLOv11  
🚀 **Personal Research Project (Independent Study)**  
📍 *Exploring structural compatibility, scaling behavior, and performance trade-offs across YOLO architectures.*

---

## 🎯 Objective
This project investigates how **interchanging YOLO backbones and heads** affects detection performance, computational cost, and architectural stability.

> Rather than treating YOLO architectures as immutable, this experiment approaches them **as modular systems** — where *backbone–neck–head components* can be recombined to discover hybrid efficiency.

---

## 🧩 Model Variants

| Variant    | Backbone | Head    | 
| :--------- | :------- | :------ |
| **v8**     | YOLOv8   | YOLOv8  | 
| **v10**    | YOLOv10  | YOLOv10 | 
| **v11**    | YOLOv11  | YOLOv11 | |
| **v8v10**  | YOLOv8   | YOLOv10 | |
| **v10v11** | YOLOv10  | YOLOv11 | |
| **v11v8**  | YOLOv11  | YOLOv8  | |

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
text
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

> Hybridization can reduce parameter count and FLOPs while maintaining comparable feature depth.

---

## 📊 Performance Summary

### Validation mAP (AgriPest Valid – 1,095 images)

| Model   | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :------ | ------------: | ---------: | ----: | -------: |
| **v8**  | 0.676 | 0.597 | **0.660** | **0.416** |
| **v10** | 0.634 | 0.485 | 0.586 | 0.380 |
| **v11** | 0.629 | 0.573 | 0.627 | 0.395 |

🧩 **Preliminary insight:**
- YOLOv8 performs strongest on small insects (dense features).  
- YOLOv11 offers lowest FLOPs → fastest inference.  
- Hybrid **v11-backbone + v8-head** may offer optimal efficiency–accuracy trade-off *(training in progress).*

---

## ⏱️ Training & Inference Time

| Model   | Training Time | Inference (ms/image) |
| :------ | -------------: | -------------------: |
| YOLOv8  | 531 m 52.5 s  | 36.9 |
| YOLOv10 | 699 m 12.4 s  | 17.2 |
| YOLOv11 | 564 m 51.0 s  | 14.4 |

> Despite higher training cost, **v11** achieves fastest inference on CPU, aligning with its lower FLOPs count.

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

(Full YAMLs available in /models/custom/)

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