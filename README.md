# EAVAE: Uncertainty in latent representations of VAEs - natural images domain

This repository contains the code for the natural images domains and divisive normalization experiments presented in the manuscript *Uncertainty in latent representations of variational autoencoders optimized for visual tasks*. This allows for evaluating the trained models presented in the text, or training new models in this regime. 

The code for the rest of the paper can be found at https://github.com/josefina-catoni/EA-VAE-uncertainty-in-latent-representations.

---

## Installation

1. Clone the repository:

```
git clone https://github.com/martosdomo/EA-VAE-natural-domain.git
```

---

2. Create and activate a Virtual Environment
Navigate to the directory where you want to create your virtual environment:

```
cd /path/to/your/project/EA-VAE-natural-domain
```

Create a virtual environment with conda (Download and install first):

```
conda create --name my_env python=3.11.4
```

Replace myenv with the name you want for the virtual environment.

**Activate the Virtual Environment**

On Linux:
```
conda activate myenv
```

You should see the environment name (myenv) in your terminal prompt, indicating the virtual environment is active.

3. Install Packages
After activating the environment, you can

Install Pytorch (https://pytorch.org/)

Install required libraries

```
pip install -r requirements.txt
```

---

## Project Structure

```
├── data                            # Datasets, dataset loaders
├── divisivenorm                    # Divisive normalization analysis and visualization
├── eval_data                       # Directory for saving data
├── evaluate                        # Evaluation and visualization of trained models
|   ├── aux_functions.py            # Auxiliary functions
|   ├── evaluate.py                 # Script for evaluating the model
|   ├── figures_natural.ipynb       # Notebook for visualizations
├── experiments                     # Local checkpoints, logs. Trained manuscript models.
├── hvae_backbone                   # Core framework components
├── models                          # Model definitions
├── scripts                         # Main scripts
│   ├── test.py                     # Test script
│   ├── train.py                    # Training script
├── config.py                       # Choose your model file here!
```

---

## Choosing a model

Models are defined in a single file, in the `models` folder. The models presented in the manuscript are included with the original hyperparameters as `StandardVAE.py`, `EAVAE_softlaplace.py` and `EAVAE_lognormal.py`. You can use these models, or train your own.

To **train your model**, you need to:

1. Configure the training **hyperparameters** in your model file. See the comments in your model file for more information.
2. Import your model in the `config.py` file.
   ```python
   def get_hparams():
      # SET WHICH params TO USE HERE
      # |    |    |    |    |    |
      # v    v    v    v    v    v
      import models.YourModel as params
      ...
   ```
3. Run `train.py` in the `scripts` directory.
   ```bash
   python scripts/train.py
   ```
   This will start training your model with the given hyperparameters.
   The training set of the dataset will be used for training and the validation set for validation.

**Training logs**  

When you start training, a new directory will be created in the `experiments` directory with the name of your model.
In this directory, another directory will be created timestamped with the current date and time. The logs and the latest checkpoints will be saved here.

**Training from a checkpoint**  

To train your model from a checkpoint, you need to set the `load_from_train` parameter in the `log_params` of your model file to the path to the checkpoint you want to start from.


---
## Evaluation

To **evaluate your model**, you need to:

1. Set the `load_from_eval` parameter in the `log_params` of your model file to the path to the checkpoint of the model you want to evaluate. By default, the checkpoints of the trained manuscript models are given.
2. Configure the evaluation **hyperparameters** in your model file. See the comments in your model file for more information. 
3. Import your model in the `config.py` file. This is the same as for training.
4. Run `evaluate/evaluate.py` 
   ```bash
   python evaluate/evaluate.py
   ```
   This will execute all the presented analyses and save the files in the folder `eval_data` under the model's name.

To **plot the figures**:

1. Open `evaluate/figures_natural.ipynb`.
2. Define your model names and other settings in the config dictiionary.
3. Run the notebook. Resulting figs will be saved by default.

---
## Divisive normalization experiments

To perform the DN experiments, you need to evaluate your models as seen above. This will save data required for DN in `eval_data/DN`. To perform the extra analysis necessary for these experiments:

1. In `divisivenorm/fit_schwartz2001` and in `divisivenorm/plot_bowties` set your model names in *config.py*.
2. Run `divisivenorm/fit_schwartz2001/fit_schwartz2001.py` and `divisivenorm/plot_bowties/plot_bowties.py`. This will save necessary files to `divisivenorm/results`. Note that the former includes a longer training process. 

Example data of StandardVAE and EAVAE_lognormal is included in the repository in `divisivenorm/results`. You can also use this for plotting.

To plot the figures:

1. Open `divisivenorm/figures_divisivenorm.ipynb`.
2. Set the models you want to evaluate (`standardvae_modelname` and `eavae_modelname` variables). For Panel B & C, data should be present in `divisivenorm/results` for these models.
3. Run the notebook. Note that some cells might take longer to run.
