# Installation & License

## Installation

:::{warning}
FAIRChem V2 is a major breaking change from V1 and is not compatible with previous pretrained models. If you need the old V1 code, install version 1 with `pip install fairchem-core==1.10`.
:::

To install `fairchem-core` you will need to setup the `fairchem-core` environment. We support either pip or uv. Conda is no longer supported and has also been dropped by pytorch itself. Note you can still create environments with conda and use pip to install the packages.

:::{tip}
We recommend installing fairchem inside a virtual environment instead of directly onto your system.
:::

### Step 1: Create a virtual environment

```bash
virtualenv -p python3.12 fairchem
source fairchem/bin/activate
```

### Step 2: Install the package

```bash
pip install fairchem-core
```

:::{admonition} For developers contributing to fairchem
:class: dropdown

Clone the repo and install in editable mode:

```bash
git clone git@github.com:facebookresearch/fairchem.git
cd fairchem
pip install -e src/packages/fairchem-core[dev]
```
:::

:::{note}
In V2, we removed all dependencies on 3rd party libraries such as torch-geometric, pyg, torch-scatter, torch-sparse etc that made installation difficult. So no additional steps are required!
:::

### GPU support (mlipx / `mlipx setup`)

The right PyTorch build depends on your GPU's compute capability. The mlipx CLI can detect your GPU and print the exact install command — it uses `nvidia-smi`, so it works **before** PyTorch is installed:

```bash
uv run mlipx setup      # detect GPU + print the exact torch install command
uv run mlipx doctor     # verify after install
```

Supported floor: **Maxwell (GTX 900 series, e.g. GTX 960)**. Kepler (GTX 700/600) is not supported (no prebuilt PyTorch wheel).

| GPU family | CC | Recommended torch |
|---|---|---|
| Maxwell (GTX 750/9xx) | sm_50/52 | `torch==2.6.0+cu124` |
| Pascal (GTX 10xx, P104-100) | sm_60/61 | `torch==2.6.0+cu124` |
| Volta–Hopper (V100…H100, RTX 20/30/40) | sm_70–90 | `torch==2.6.0+cu124` |
| Blackwell (RTX 50) | sm_100/120 | `torch==2.8.0+cu128` |
| Kepler (GTX 700/600) | sm_30/37 | not supported |

PyTorch 2.7+ dropped `sm_50`/`sm_60` from its prebuilt CUDA wheels, so Maxwell/Pascal cards must stay on `torch 2.6.0+cu124` (its `sm_50`/`sm_60` kernels are binary-compatible with `sm_52`/`sm_61`). This workspace pins `torch==2.6.0+cu124` by default, so `uv sync` works out of the box for Maxwell–Hopper; only Blackwell needs the `cu128` override. If the download fails, enable a proxy first: `clashctl on`.


## Subpackages

In addition to `fairchem-core`, there are related packages for specialized tasks or applications. Each can be installed with `pip` or `uv` just like `fairchem-core`:

### Data Packages

Utilities for generating input configurations and working with specific datasets:

::::{grid} 1 2 3 3

:::{card} fairchem-data-oc
Code for generating adsorbate-catalyst input configurations
:::

:::{card} fairchem-data-omat
Code for generating OMat24 input configurations and VASP input sets
:::

:::{card} fairchem-data-omc
Code for generating OMC (Molecular Crystals) VASP inputs
:::

:::{card} fairchem-data-omol
Code for generating OMOL input configurations
:::

:::{card} fairchem-data-odac
Code for ODAC MOF configurations and VASP input sets for direct air capture
:::

::::

### Application Packages

Higher-level applications built on top of FAIRChem models:

::::{grid} 1 2 3 3

:::{card} fairchem-applications-adsorbml
Module for calculating minimum adsorption energies
:::

:::{card} fairchem-applications-cattsunami
Accelerating transition state energy calculations with pre-trained GNNs
:::

:::{card} fairchem-applications-fastcsp
Accelerated molecular crystal structure prediction with UMA
:::

:::{card} fairchem-applications-ocx
Bridging experiments to computational models
:::

::::

### Integration & Demo Packages

Tools for integrating with other software or demo APIs:

::::{grid} 1 2 3 3

:::{card} fairchem-lammps
Use FAIRChem models with LAMMPS for large-scale MD simulations
:::

:::{card} fairchem-demo-ocpapi
Python client library for the Open Catalyst Demo API
:::

::::


## Access to gated models on HuggingFace

To access gated models like UMA, you need to get a HuggingFace account and request access to the UMA models.

:::{admonition} HuggingFace Setup Steps
:class: tip

1. Get and login to your HuggingFace account
2. Request access to <https://huggingface.co/facebook/UMA>
3. Create a HuggingFace token at <https://huggingface.co/settings/tokens/> with the permission "Read access to contents of all public gated repos you can access"
4. Add the token as an environment variable using `huggingface-cli login` or by setting the `HF_TOKEN` environment variable
:::

## License

### Repository software

The software in this repo is licensed under an MIT license unless otherwise specified.

:::{admonition} MIT License
:class: dropdown

```text
MIT License

Copyright (c) Meta, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
:::

### Terms of use & privacy policy

Please read the following [Terms of Use](https://opensource.fb.com/legal/terms) and
[Privacy Policy](https://opensource.fb.com/legal/privacy) covering usage of `fairchem` software and models.

### Model checkpoints and datasets

Please check each dataset and model for their own licenses.
