# Advanced Computer Vision notebooks, PyTorch edition

PyTorch ports of all 16 TensorFlow/Keras notebooks in `../cv` (Coursera *Advanced Computer Vision with TensorFlow*, course 3 of the TensorFlow: Advanced Techniques specialization).

Every notebook keeps the structure, prose and exercises of the original; only the framework changed. Graded assignments keep their `None` / `# YOUR CODE HERE` placeholders, with the prompts rewritten for PyTorch. The Coursera autograders expect Keras artifacts, so the PyTorch assignments cannot be submitted for grading.

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/) (Python 3.12, PyTorch 2.14, torchvision 0.29).

```bash
cd cv_pytorch
uv sync                 # creates .venv and installs everything from uv.lock
uv run jupyter lab      # or: uv run jupyter notebook
```

Training uses CUDA when available, Apple's MPS backend on Apple Silicon Macs, and the CPU otherwise. Datasets and pretrained weights are downloaded into `cv_pytorch/data/` (or torchvision's cache) the first time a notebook needs them; the notebooks skip downloads that are already present.

## Start here

**`C3_W0_Prerequisites_PyTorch_CV.ipynb`** has no TensorFlow counterpart. It covers the things the course assumes but never states, because the original was written in TensorFlow: the channels-first image layout, what each `transforms` operation does to a picture, normalizing and un-normalizing for display, `unsqueeze`/`squeeze`/`permute`/`reshape`, how `Dataset` and `DataLoader` replace `tf.data`, convolution and transposed-convolution arithmetic, why models return logits instead of probabilities, and the anatomy of a training loop. It then explores every dataset used in this project: class balance, image size spread, mask label meanings, and the class imbalance that makes per-class IOU the right metric for the segmentation labs.

Its final section is nine hands-on tasks that answer *why* each routine step is there, by removing it and showing what breaks. Skipping `Resize` makes batching fail outright; skipping `Normalize` drops a pretrained classifier from 0.83 confidence to 0.19 and changes its answer; `shuffle=False` on class-sorted data lands at 50% accuracy, pure chance; forgetting `zero_grad()` accumulates gradients 3, 6, 9; computing cross entropy from probabilities instead of logits returns literal infinity; and a segmentation model that predicts only background scores 94.4% pixel accuracy with 0.000 IOU on every digit.

## Notebooks

| Notebook | Original | What changed |
| --- | --- | --- |
| `C3_W1_Lab_1_transfer_learning_cats_dogs` | same name | `ImageDataGenerator` -> torchvision `transforms` + `ImageFolder`; Keras InceptionV3 cut at `mixed7` -> torchvision `inception_v3` cut at `Mixed_6e`. |
| `C3_W1_Lab_2_Transfer_Learning_CIFAR_10` | same name | Keras ResNet50 -> torchvision `resnet50`; `preprocess_input` -> ImageNet normalization. |
| `C3_W1_Lab_3_Object_Localization` | same name | Synthetic MNIST-on-canvas `tf.data` pipeline -> `Dataset`; two-output Functional model -> `nn.Module` returning two tensors. |
| `C3W1_Assignment` | same name | Caltech Birds TFRecords read with the pure-Python `tfrecord` package (no TensorFlow needed); MobileNetV2 from torchvision; explicit training loop is Exercise 7. |
| `C3_W2_Lab_1_Simple_Object_Detection` | same name | TF Hub Faster R-CNN Inception ResNet V2 -> torchvision `fasterrcnn_resnet50_fpn_v2`, so classes are COCO's; boxes come back in pixels as `[xmin, ymin, xmax, ymax]` rather than normalized `[ymin, xmin, ymax, xmax]`. |
| `C3_W2_Lab_2_Object_Detection` | same name | TF Hub Open Images detectors -> torchvision COCO detectors (`ssdlite320_mobilenet_v3_large`, `fasterrcnn_resnet50_fpn_v2`). |
| `C3W2_Assignment` | same name | TF Object Detection API RetinaNet -> torchvision `retinanet_resnet50_fpn`; config files -> builder kwargs and weights metadata; `tf.train.Checkpoint` -> selective `load_state_dict`. The Colab-only box annotation tool is replaced by the provided boxes. |
| `C3_W3_Lab_1_VGG16_FCN8_CamVid` | same name | VGG16 encoder + FCN-8 decoder as `nn.Module`s, pretrained VGG weights copied from torchvision; label maps stay integer class ids (no one-hot). |
| `C3_W3_Lab_2_OxfordPets_UNet` | same name | Oxford-IIIT Pet from `torchvision.datasets.OxfordIIITPet` instead of TFDS; the UNet blocks are `nn.Module`s; Keras `Conv2DTranspose(strides=2, padding='same')` -> `ConvTranspose2d(stride=2, padding=1, output_padding=1)`; masks resized with nearest neighbour so no class ids are invented. |
| `C3_W3_Lab_3_Mask_RCNN_ImageSegmentation` | same name | TF Hub Mask R-CNN + Object Detection API utilities -> torchvision `maskrcnn_resnet50_fpn_v2` + `torchvision.utils` drawing. |
| `C3W3_Assignment` | `Copy_of_C3W3_Assignment_Solution` | M2NIST FCN-8 assignment; the exercises build `nn.Module`s instead of Keras layers. |
| `C3_W4_Lab_1_FashionMNIST_CAM` | same name | CAM model built by slicing the `nn.Sequential`; features moved to channels-last so the CAM math is unchanged. |
| `C3_W4_Lab_2_CatsDogs_CAM` | same name | `tfds` cats_vs_dogs -> the same Microsoft image archive through a small `Dataset` (with the corrupt files skipped). |
| `C3_W4_Lab_3_Saliency` | same name | TF Hub Inception V3 (1001 classes) -> torchvision `inception_v3` (1000 classes, so every ImageNet id shifts down by one); gradients via `torch.autograd.grad`; ImageNet normalization wrapped as a layer so the saliency map lines up with the visible pixels. |
| `C3_W4_Lab_4_GradCam` | same name | Intermediate activations and Grad-CAM via forward hooks and `torch.autograd.grad`. |
| `C3W4_Assignment` | same name | Saliency-map assignment; the provided Keras `.h5` weights are loaded into the PyTorch model by a helper (`load_keras_weights`). |

## Code style

- **Every function and class carries a docstring**, with `Args:` and `Returns:` naming tensor shapes wherever shapes are not obvious.
- **No implicit globals.** Functions take what they use. `run_epoch`, `evaluate`, `predict`, `get_CAM`, `show_cam` and friends receive `model`, `loss_fn`, `optimizer` and `device` as parameters rather than reading them from the notebook scope, so a reader can tell what a function depends on from its signature alone. Module-level constants (`BATCH_SIZE`, `class_names`, `colors`) stay as constants, and helper functions are still called by name.
- **Shape comments** are terse and inline, in the form `# shape: xb (N, 3, 150, 150) | yb (N,) -> (N, 1)`.

## Conventions used in the ports

- Models output raw logits; `nn.CrossEntropyLoss` / `nn.BCEWithLogitsLoss` apply the softmax/sigmoid, and the notebooks apply it explicitly when probabilities are needed.
- Keras' `RMSprop` defaults (`rho=0.9`, `epsilon=1e-7`) are passed explicitly as `alpha`/`eps`, because PyTorch's defaults take much larger first steps.
- Pretrained torchvision weights expect ImageNet mean/std normalization (InceptionV3 expects `[-1, 1]`), so the preprocessing differs from the Keras `preprocess_input` functions.
- `DataLoader` workers on macOS use the `fork` start method (`multiprocessing_context=MP_CONTEXT`), because `spawn` workers cannot see `Dataset` classes defined inside a notebook.

## Training time

The epoch counts are the ones from the original notebooks (e.g. 50 epochs for the birds assignment, 170 for CamVid, 25 for the cats-vs-dogs CAM lab). They were written for a Colab GPU; on a laptop CPU they take hours, so lower `EPOCHS` for a first run. Every notebook was verified end to end with reduced epochs on an Apple Silicon Mac (MPS).
