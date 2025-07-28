### Environment Dependencies & Installation

#### Python Version
- **Python 3.8+** (required for PyTorch 2.0+ and Ultralytics)

---

#### Core Dependencies
Install via `pip`:
```bash
pip install torch torchvision ultralytics opencv-python numpy pillow matplotlib
```

Install via `conda`:
```bash
conda install -c pytorch -c conda-forge pytorch torchvision
conda install -c conda-forge ultralytics opencv pillow matplotlib numpy
```

---

#### Key Package Versions
| Package      | Minimum Version | Notes                                  |
|--------------|-----------------|----------------------------------------|
| PyTorch      | 2.0+            | GPU acceleration with CUDA 11.8        |
| Ultralytics  | 8.0.0+          | YOLOv8 framework                      |
| OpenCV       | 4.5.0+          | Image processing                       |
| TorchVision  | 0.15.0+         | MNIST dataset loader                   |
| Matplotlib   | 3.5.0+          | Visualization                          |

---

#### GPU Acceleration Setup
1. **NVIDIA Drivers**  
   - Requires driver version ≥ 525.60.13 for CUDA 11.8
   - Verify with `nvidia-smi`

2. **CUDA Toolkit**  
   Install CUDA 11.8:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```

3. **cuDNN**  
   Download v8.7.0 for CUDA 11.x from [NVIDIA Developer](https://developer.nvidia.com/cudnn)

---

#### English Summary

This project implements **YOLOv8 for MNIST digit detection** using PyTorch. Key dependencies include:
1. **Ultralytics YOLOv8** - For object detection pipeline
2. **PyTorch with CUDA** - GPU-accelerated training
3. **TorchVision** - MNIST dataset handling
4. **OpenCV** - Image processing and test image generation
5. **Matplotlib** - Visualization of results

The script:
- Creates YOLO-formatted datasets from MNIST
- Trains YOLOv8l (large model) for 128 epochs
- Validates using mAP@0.5 and precision/recall metrics
- Generates multi-digit test images
- Runs predictions with visualization

**Hardware Note**: Optimized for NVIDIA GPUs (e.g., RTX 4060 8GB). Batch sizes are configured for 8GB VRAM. For CPU-only mode, set `device='cpu'` in training arguments.
