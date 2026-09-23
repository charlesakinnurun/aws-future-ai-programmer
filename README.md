![](/assets/aws.png)
# CNN-Based Dog Breed Image Classification

A Python CLI that compares three pretrained CNN architectures (AlexNet, VGG-16, ResNet-18) to classify pet images as **dog** or **not-a-dog** and to identify the dog **breed**, as part of the Udacity *AI Programming with Python* "Classify Pet Images" capstone project.

## Overview

A citywide dog show requires that every registered contestant submit a photo of their dog. Some participants try to register pets that are not dogs. The organizing committee needs a reliable way to confirm, from the submitted photo, whether the animal is actually a dog and—if so—which breed it is.

This project builds an end-to-end Python evaluation pipeline that uses **already-trained** convolutional neural networks (CNNs) from `torchvision`, rather than training a new model. The core contribution is the Python engineering that:

1. Extracts the ground-truth pet label from each image's filename.
2. Runs inference with each of three ImageNet-pretrained CNN architectures.
3. Compares classifier output to the true label.
4. Determines whether each image was correctly classified as `dog` / `not-a-dog` and, for dogs, whether the **breed** was correct.
5. Computes summary statistics in counts and percentages, and times the full run.

The result answers which architecture—**AlexNet, VGG, or ResNet**—works "best" for this task and how well it performs on breed identification.

> **Repo-name note:** the repository is named `aws-image-classifier-for-dog-breed`, but **no AWS services are used** anywhere in the code. The pipeline runs locally on CPU with PyTorch.

## Features

- **Three CNN architectures** benchmarked: ResNet-18, AlexNet, and VGG-16 (all ImageNet-pretrained via `torchvision`).
- **Automatic ground-truth labeling** from image filenames (e.g. `Boston_terrier_02259.jpg` → `boston terrier`).
- **Two evaluation views** over one run:
  - *is-a-dog* correctness (dog vs. not-a-dog, regardless of breed),
  - *breed* correctness (dog images whose predicted label matches the true label).
- **Summarized statistics** (counts and percentages) plus optional printouts of every misclassified dog/not-dog assignment and misclassified breed.
- **Program runtime report** in `hh:mm:ss` for every run.
- **Fault-tolerant matching**: classifier labels can contain multiple names per class (e.g. `dalmatian, coach dog, carriage dog`) and are matched by substring.
- **CLI-driven** via `argparse` with sensible defaults, so the same code runs across models and image folders.

## Tech Stack

| Category | Technology |
| -------- | ---------- |
| Language | Python 3 (bytecode artifacts indicate Python 3.14 was used) |
| Deep Learning | PyTorch + TorchVision (pretrained `resnet18`, `alexnet`, `vgg16`) |
| Image Processing | Pillow (PIL) |
| Numerical | NumPy |
| CLI | `argparse` (standard library) |<>
| Scheduling | POSIX shell scripts (`run_models_batch*.sh`) |

No web framework, API server, database, container, or cloud configuration is present in the repository, and there is no `requirements.txt` (see [Installation](#installation)).

## Project Architecture

The pipeline is a single-pass evaluation loop. The same code path is triggered for each model:

```mermaid
flowchart LR
    A[Images\npet_images/ or uploaded_images/] --> B[Extract pet label from filename\nget_pet_labels.py]
    B --> C[CNN inference per image\nclassifier.py - resnet | alexnet | vgg]
    C --> D[Compare classifier vs pet label\nclassify_images.py]
    D --> E[Flag is-a-dog for pet & classifier labels\nadjust_results4_isadog.py]
    E --> F[Compute counts & percentages\ncalculates_results_stats.py]
    F --> G[Print summary + misclassifications\nprint_results.py]
    G --> H[Report total runtime\ntime module]
```

Data flow per image inside the results dictionary (key = filename):

| Index | Content |
| ----: | ------- |
| 0 | Pet label parsed from filename |
| 1 | Classifier label (lowercased, stripped) |
| 2 | `1` = classifier label contains pet label, else `0` (breed match) |
| 3 | `1` = pet label is a dog name, else `0` |
| 4 | `1` = classifier label is a dog name, else `0` |

## Project Structure

```text
aws-image-classifier-for-dog-breed/
├── data/                              # All project code and data live here
│   ├── check_images.py                # MAIN entry point: end-to-end pipeline
│   ├── classifier.py                  # Pretrained-CNN wrapper returning ImageNet labels
│   ├── test_classifier.py             # Minimal demo of classifier() usage
│   ├── get_input_args.py              # argparse: --dir, --arch, --dogfile
│   ├── get_pet_labels.py              # Ground-truth label from filename
│   ├── classify_images.py             # Run inference + breed-match comparison
│   ├── adjust_results4_isadog.py      # is-a-dog flags (indices 3 & 4)
│   ├── calculates_results_stats.py    # Counts & percentage statistics
│   ├── print_results.py               # Summary + optional misclassification printouts
│   ├── print_functions_for_lab_checks.py # Self-check helpers used during development
│   ├── *_hints.py                     # Course "hints" reference implementations
│   ├── dognames.txt                   # 224 dog-name entries (incl. alternate names)
│   ├── imagenet1000_clsid_to_human.txt# ImageNet 1000-class index -> readable label
│   ├── run_models_batch.sh            # Run all 3 models on pet_images/
│   ├── run_models_batch_uploaded.sh   # Run all 3 models on uploaded_images/
│   ├── check_images.txt               # Worksheet for the uploaded-image exercise (blank)
│   ├── alexnet_uploaded-images.txt    # Captured console output (failed run, see Results)
│   ├── resnet_uploaded-images.txt     # Captured console output (failed run, see Results)
│   ├── vgg_uploaded-images.txt        # Captured console output (failed run, see Results)
│   ├── pet_images/                    # 40 labeled images (30 dogs, 10 non-dogs)
│   └── uploaded_images/               # 4 user-uploaded sample images
├── project-workspace-<step>/          # 8 step folders linking to the exercise files
│   └── Readme.md
├── .gitignore                         # Ignores data/__pycache__
└── README.md
```

> The repository also contains untracked root-level copies of `alexnet/resnet/vgg_uploaded-images.txt`; they duplicate the versions under `data/` and are not part of the tracked history.

Key source files:

- `data/check_images.py:41-126` — orchestrates the whole pipeline and times the run.
- `data/classifier.py` — loads `resnet18`, `alexnet`, `vgg16` with `pretrained=True`, applies the standard ImageNet preprocessing (Resize 256 → CenterCrop 224 → normalize with ImageNet mean/std), runs the model in `eval()` mode, and maps the argmax logit to a human-readable label via `imagenet1000_clsid_to_human.txt`.
- `data/get_input_args.py` — CLI arguments with defaults: `--dir pet_images/`, `--arch vgg`, `--dogfile dognames.txt`.

## Dataset

**Primary evaluation set — `data/pet_images/` (40 images):**

- **Ground truth is encoded in the filename**, e.g. `Golden_retriever_05223.jpg` → *golden retriever*. The label is derived by splitting the filename on `_` and keeping only alphabetic words, lowercased.
- **30 dog images** spanning 16 breeds: Basenji, Basset hound, Beagle, Boston terrier, Boxer, Cocker spaniel, Collie, Dalmatian, German shepherd dog, German shorthaired pointer, Golden retriever, Great dane, Great pyrenees, Miniature schnauzer, Poodle, Saint Bernard.
- **10 non-dog images**: rabbit, cats (3), fox squirrel, geckos (2), great horned owl, polar bear, skunk.

**Secondary set — `data/uploaded_images/` (4 images):**
Two dog photos (`Dog_01.jpg.webp`, `Dog_02.jpg`) and two non-dog photos (`Cat_01.jpg.jpg`, `Coffee_mug_01.jpg.jpg`) simulating real contestant submissions.

**Supporting data:**
- `data/dognames.txt` — 224 entries; one dog name (or comma-separated alternate names) per line, used to decide *is-a-dog* for both pet and classifier labels.
- `data/imagenet1000_clsid_to_human.txt` — the full 1,000-class ImageNet index-to-label mapping (1,000 lines) used to translate a model's argmax class index into text.

**Split:** there is no train/validation/test split. The models are pretrained and used strictly for inference; all images are treated as an evaluation set.

**Preprocessing (inside `classifier.py`):** resize to 256 → center-crop to 224×224 → tensor → normalize with ImageNet channel means/stds `[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]`.

## Machine Learning Approach

- **Problem formulation.** This is an *image classification use-case* built on top of an existing classifier — the project intentionally focuses on Python pipeline engineering, not on training.
- **Models.** Three ImageNet-pretrained CNNs from TorchVision are compared: `resnet18`, `alexnet`, and `vgg16`. Pretrained weights let the pipeline run with no training and no labeled data; the trade-off is that breed resolution is limited to the (roughly 120) dog classes ImageNet contains.
- **Inference.** Each image is resized/normalized, passed through the model in `eval()` mode, and the argmax of the 1,000-class softmax output is mapped to its human-readable label.
- **Breed match.** A pet label matches a classifier label if the pet label is a *substring* of the (lowercased) classifier label, which also handles multi-name classes such as `dalmatian, coach dog, carriage dog`.
- **is-a-dog decision.** Both the pet label and the classifier label are looked up against the keys of `dognames.txt`; each gets a `1`/`0` flag. Correct dog/not-dog classification is then compared independently of breed correctness.
- **Evaluation metrics** (per run, from `calculates_results_stats.py`):
  - `pct_match` — % of images where the classifier label contains the pet label (all images).
  - `pct_correct_dogs` — % of dog images classified as a dog.
  - `pct_correct_breed` — % of dog images with the correct breed (labeled true dog **and** match).
  - `pct_correct_notdogs` — % of non-dog images classified as not-a-dog.
  - Corresponding counts: `n_images`, `n_dogs_img`, `n_notdogs_img`, `n_match`, `n_correct_dogs`, `n_correct_notdogs`, `n_correct_breed`.
- **Hyperparameters.** The models use their default pretrained checkpoints; no hyperparameters are tuned (no training occurs).
- **Inference workflow.** One CLI invocation classifies every image in the target folder under a single architecture and prints the full summary. Runtime for the whole run (model load + all predictions) is timed and printed in `hh:mm:ss`.

## Model Performance

**No performance results are recorded in the repository.**

The three captured output files (`data/*_uploaded-images.txt`) do **not** contain model results — each contains a Python traceback from a run that failed on this machine:

```text
ModuleNotFoundError: No module named 'PIL'
```

i.e. the pipeline was executed before dependencies were installed, so no accuracy numbers, breed-accuracy values, or runtime figures are available anywhere in the repo. The worksheet `data/check_images.txt` with the four analysis questions for the uploaded images is also unanswered.

Once dependencies are installed (see below), results for each architecture can be regenerated and the summary section updated.

## Installation

There is **no `requirements.txt`** in the repository. Dependencies are inferred from the imports in `data/classifier.py`:

- `torch` + `torchvision` (model zoo and transforms)
- `pillow` (image I/O)
- `numpy` (logit → argmax handling)

```bash
git clone https://github.com/charlesakinnurun/aws-image-classifier-for-dog-breed.git
cd aws-image-classifier-for-dog-breed

python -m venv .venv
```

Activate the environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies (your `torch` install string may differ by OS/accelerator — see https://pytorch.org/get-started):

```bash
pip install torch torchvision pillow numpy
```

On first use, TorchVision downloads the pretrained ImageNet weights automatically (network connection required). The repository's `.pyc` artifacts (`cpython-314`) indicate the code was last run under Python 3.14, but any modern Python 3.x should work.

## Usage

All scripts use relative paths, so run them **from inside the `data/` directory**:

```bash
cd data
```

### Demo the classifier function

```bash
python test_classifier.py
```

Classifies `pet_images/Collie_03797.jpg` with VGG and prints the predicted label.

### Run a single model on the pet images

```bash
python check_images.py --dir pet_images/ --arch resnet  --dogfile dognames.txt
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt
python check_images.py --dir pet_images/ --arch vgg    --dogfile dognames.txt
```

`--arch` is restricted to `resnet`, `alexnet`, or `vgg`; `--dir` defaults to `pet_images/`, `--arch` to `vgg`, and `--dogfile` to `dognames.txt`.

### Run all models via the batch script

```bash
sh run_models_batch.sh            # pet_images → resnet/alexnet/vgg_pet-images.txt
sh run_models_batch_uploaded.sh   # uploaded_images → *_uploaded-images.txt
```

### Classify the "contestant submission" images

```bash
python check_images.py --dir uploaded_images/ --arch vgg --dogfile dognames.txt
```

### Output

Every run prints:

- command-line arguments used,
- the pet-label dictionary sanity check,
- a per-architecture **Results Summary** with counts and percentages,
- `** Incorrect Dog/Not-Dog Assignments **` and `** Misclassified Breeds **` listings (both enabled in `check_images.py`),
- total elapsed runtime in `hh:mm:ss`.

## Configuration

There are **no environment variables and no configuration files**. The only knobs are the three CLI arguments documented above (`--dir`, `--arch`, `--dogfile`), all with defaults.

## Examples

- **`data/test_classifier.py`** — the minimal "hello world" showing how `classifier(img_path, model)` returns a human-readable label string.
- **`data/pet_images/`** — 40 real image samples that exercise every part of the pipeline and show how filenames encode ground truth.
- **`data/check_images.txt`** — a 4-question analysis worksheet for the uploaded-images run (Dog_01 breed agreement across models, Dog_01 vs Dog_02 agreement, and non-dog misclassifications). The answers are blank in the current repo.
- **`data/run_models_batch_uploaded.sh`** — demonstrates how to drive the tool over the simulated contestant submissions and capture each model's output to a file.

## Results

Beyond the (failing-run) output files described in [Model Performance](#model-performance), no verified experiment results exist in the repository. The statistics layer is in place to produce, per architecture:

| Statistic | Definition |
| --------- | ---------- |
| `pct_match` | % images classified with a matching label |
| `pct_correct_dogs` | % dog images classified as a dog |
| `pct_correct_breed` | % dog images with correct breed |
| `pct_correct_notdogs` | % non-dog images classified as not-a-dog |
| run time | total program runtime (`hh:mm:ss`) |

These can be used to fill the Model Performance table once a run succeeds.

## Deployment

No deployment configuration exists in the repository — no Dockerfile, no CI workflows, no cloud or hosting setup, and no application/UI layer. This is a local, command-line evaluation tool.

## Testing

There is **no automated test suite** (no `pytest`/`unittest` files, no test runner config).

- `data/print_functions_for_lab_checks.py` provides development-time self-check functions (`check_command_line_arguments`, `check_creating_pet_image_labels`, `check_classifying_images`, `check_classifying_labels_as_dogs`, `check_calculating_results`) that are invoked inside `check_images.py` to validate each stage of the pipeline during a normal run.
- `data/test_classifier.py` serves as a manual smoke test of `classifier()`, not as a formal test.

So the effective "test command" is to run the pipeline end-to-end and inspect the printed checks:

```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```

## Limitations

- **No recorded results.** The only captured run in the repo failed with `ModuleNotFoundError: No module named 'PIL'`; no metrics exist yet.
- **No training/fine-tuning.** Breed accuracy is bounded by the pretrained ImageNet models; certain similar breeds (e.g. Great Pyrenees vs. Kuvasz, Beagle vs. Walker Hound) are expected to be confused.
- **Ground truth comes from filenames.** Labels are assumed correct and depend on the naming convention (lowercased alphabetic words separated by `_`). The code also skips any file that does not end in `.jpg`, `.jpeg`, or `.png`.
- **Substring matching.** A breed "match" is a substring containment test (`pet_label in classifier_label`), so it can only resolve the *first* name listed for multi-name classes and can be fooled by overlapping names.
- **Exact-string is-a-dog lookup.** `dognames.txt` is matched by whole string. Because it ends with the generic entry `dog`, the semantics of an is-a-dog match depend on the exact classifier label string.
- **Single-image inference on CPU.** Each image is classified one at a time; the whole batch is sequential and the elapsed time includes loading the model weights.
- **Relative-path coupling.** `classifier.py` and the pipeline use relative paths (`imagenet1000_clsid_to_human.txt`, `images_dir + key`), so commands must be run from `data/`.
- **No packaging or dependencies file.** Without a `requirements.txt` or lockfile, environment reproducibility is manual.
- **Not reproducible results yet / no CI.** No automated tests, experiment tracking, or logging.

## Future Improvements

- **Run and record the three benchmarks**, then populate the Model Performance table with real `pct_*` values and per-model timings.
- **Add `requirements.txt`** with pinned versions and a `README` install step that works on CPU and GPU.
- **Automated tests**: unit tests for `calculates_results_stats`, `adjust_results4_isadog`, and label parsing; parametrized integration test over the three architectures; CI via GitHub Actions.
- **Timing per phase**: separate model-load time from per-image inference so architecture runtimes are comparable.
- **Fine-tune a CNN head** (or use transfer learning / embeddings) on a larger dog-breed dataset to improve breed accuracy beyond ImageNet's dog classes.
- **Batch + GPU inference**, larger seeds of dog images, and data augmentation for robustness.
- **Structured logging and config** (e.g. YAML) to make runs reproducible and auditable.
- **Optional serving layer** (FastAPI/Streamlit) to expose the classifier as a simple submission-verification service.
- **Confusion-matrix output** for the dog classes and drift checks if this becomes a production registration tool.

## License

- **Code** in this repository is MIT licensed - see `LICENSE`.

<!-- ## Author

Charles Akinnurun (attribution in source-file headers; author of this repository per the GitHub remote).-->