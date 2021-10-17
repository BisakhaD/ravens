# Transporter Networks for Shelf Placement of FMCG Objects

This repository is developed and maintained by Bisakha Das, and submitted in partial fulfilment of the requirements for her Degree of Bachelor of Computer Science Engineering of the Nanyang Technological University 

This is the reference repository for the opriginal Transporter Networks paper, developed by [Andy Zeng](https://andyzeng.github.io/), [Pete Florence](http://www.peteflorence.com/), [Daniel Seita](https://people.eecs.berkeley.edu/~seita/), [Jonathan Tompson](https://jonathantompson.github.io/), and [Ayzaan Wahid](https://www.linkedin.com/in/ayzaan-wahid-21676148/): Transporter Networks: Rearranging the Visual World for Robotic Manipulation
[Project Website](https://transporternets.github.io/)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://arxiv.org/pdf/2010.14406.pdf)&nbsp;&nbsp;•&nbsp;&nbsp;

**Abstract.** 
Versatile grasping is one of the most basic forms of robotic manipulation. Versatile grasping's purpose is to gain great autonomy in dexterous manipulation tasks in an unstructured environment. An example of such an unstructured environment with scope of versatile pick and placement options is the task of shelf-placement, requiring high and low levels of perceptual reasoning. In this paper, we chose to extend the novel model structure of Transporter Networks beyond tabletop actions performed on extruded 2-dimensional solid objects utilizing 3 Degrees of Freedom (DoF). These extensions are based on the development of an agent capable of performing shelf-placement with 6-DoF movements on two distinct axis planes. This agent has been trained on a dataset of weighted Fast-Moving Consumer Goods (FMCG) objects, both un-textured and textured. Since the training was based on imitation learning, an expert agent was developed and implemented as well. The results obtained from training the 6-DoF agent on demonstrations provided by the expert agent confirm its successful extension to 6-DoF on two planes of axis with a 70% accuracy on FMCG products. The results further indicate the success of the agent on industry benchmarked untextured and textured Yale-CMU-Berkeley (YCB) objects. In addition to extending and contributing to existing research, this work also paves the way for future research with a real UR5e robot. 

## Installation

**Step 1.** Recommended: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3.7.

```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u
echo $'\nexport PATH=~/miniconda3/bin:"${PATH}"\n' >> ~/.profile  # Add Conda to PATH.
source ~/.profile
conda init
```

**Step 2.** Create and activate Conda environment, then install GCC and Python packages.

```shell
cd ~/ravens
conda create --name ravens python=3.7 -y
conda activate ravens
sudo apt-get update
sudo apt-get -y install gcc libgl1-mesa-dev
pip install -r requirements.txt
python setup.py install --user
```

**Step 3.** Recommended: install GPU acceleration with NVIDIA [CUDA](https://developer.nvidia.com/cuda-toolkit) 10.1 and [cuDNN](https://developer.nvidia.com/cudnn) 7.6.5 for Tensorflow.
```shell
./oss_scripts/install_cuda.sh  #  For Ubuntu 16.04 and 18.04.
conda install cudatoolkit==10.1.243 -y
conda install cudnn==7.6.5 -y
```

### Alternative: Pure `pip`

As an example for Ubuntu 18.04:

```shell
./oss_scipts/install_cuda.sh  #  For Ubuntu 16.04 and 18.04.
sudo apt install gcc libgl1-mesa-dev python3.8-venv
python3.8 -m venv ./venv
source ./venv/bin/activate
pip install -U pip
pip install scikit-build
pip install -r ./requirements.txt
export PYTHONPATH=${PWD}
```

## Getting Started

**Step 1.** Generate training and testing data (saved locally). Note: remove `--disp` for headless mode.

```shell
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=shelf-placing --mode=train --n=10
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=shelf-placing --mode=test --n=100
```

To run with shared memory, open a separate terminal window and run `python3 -m pybullet_utils.runServer`. Then add `--shared_memory` flag to the command above.

**Step 2.** Train the 6-DoF Transporter Networks model. Model checkpoints are saved to the `checkpoints` directory. Optional: you may exit training prematurely after 2000 iterations to skip to the next step, and later continue logging with ```shell --continue_logging True --log_ckpt 2000```.

```shell
python ravens/train.py --task=shelf-placing --agent=transporter_6d --n_demos=10
```

**Step 3.** Evaluate the 6-DoF Transporter Networks agent using the model trained for 2000 iterations. Results are saved locally into `.pkl` files.

```shell
python ravens/test.py --assets_root=./ravens/environments/assets/ --disp=True --task=shelf-placing --agent=transporter_6d --n_demos=10 --n_steps=2000
```

**Step 4.** Plot and print results.

```shell
python ravens/plot.py --disp=True --task=shelf-placing --agent=transporter_6d --n_demos=10
```

**Optional.** Track training and validation losses with Tensorboard.

```shell
python -m tensorboard.main --logdir=logs  # Open the browser to where it tells you to.
```
