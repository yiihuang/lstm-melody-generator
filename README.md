# 📊 Project Title: *Melody Generation*

## Overview

To solve the complex problem of music generation, this project employs long short-term memory (LSTM) neural networks to model musical sequences as a time series.

By analyzing temporal and melodic relationships, the model predicts the next note or melody, effectively addressing the challenge of generating music with artificial intelligence.

### Dataset

The dataset is retrieved from `http://kern.ccarh.org/` which is a library of virtual musical scores in the Humdrum **kern** data format.
Total holdings: 7,866,496 notes in 108,703 files

### Audience

Developers or researchers interested in music generation and deep learning.

## 🎵 Audio Preview

Listen to the generated melodies from our LSTM model:

### Generated Melody 1
<audio controls>
  <source src="example/Untitled_01.mp3" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

### Generated Melody 2
<audio controls>
  <source src="example/Untitled_02.mp3" type="audio/mpeg">
  Your browser does not support the audio element.
</audio>

---

## 📁 Project Structure

```
Melody Generation/
│
├── dataset/                              # Encoded songs to simple time series representation with numbers.
├── melodies/
│          ├── [0,1,3.....,1699]          # 1699 melodies encoded files.
├── file_dataset                          # One file combines all encoded melodies into one file.
├── mapping.json                          # Mapping file that maps our encoded time series text representations into numbers.
├── Dockerfile/                           # Dockerfile to run the project.
├── src/                                  # Scripts for reusable code (cleaning, modeling, etc.)
│   ├── melodygenerator.py                # Script that handles generating melody as midi file extension from the output of a trained model.
│   ├── preprocess.py                     # Script for loading melody songs and encode them into understandable timeseries text/string for midi files extension.
│   ├── train.py                          # Script for training building, compiling the model as RNN LSTM and save the output model. 
│   ├── utils.py                          # Utils helper functions.
├── models/                               # Saved trained models for reuse or deployment.
│   ├── model.h5                          # The saved trained model.
├── outputs/                              # Generated melodies.
│   ├── mel.mid                           # Our main generated melody using our trained model as midi file extension.                    
├── main.py                               # The entery point of the our project.
├── requirements.txt                      # List of Python packages needed to run the project.
├── README.md                             # Project description and usage guide (this file).
└── .gitignore                            # Ignore virtual environments, model files, etc. in Git.
└── .dockerignore                         # Ignore virtual environments, model files, etc. in docker build.

```

---

## 🔧 Setup and Installation Instructions

### Option 1. Google Colab

Using google colab is the prefered method for easier setup with GPUs.


### Option 2. Docker

#### 1. Docker Build Step

In oder to run using docker, we have to build the image first and make sure we have our GPUs CUDA runtime installed.

To build the docker image, please run the following command:

```bash
docker build -t compu-lstm .
```

#### 2. Check for GPU installation

##### Windows

**Check Prerequisites:** Ensure your system has a CUDA-capable GPU, a supported version of Windows, and a compatible version of Microsoft Visual Studio.

**Download the CUDA Toolkit:** Go to the official NVIDIA CUDA downloads page and select your platform: Windows, x86_64 architecture, and your specific Windows version.

**Run the Installer:** Download either the network installer or the full installer. Run the downloaded .exe file and follow the on-screen instructions.

**Verify Installation:** After the installation is complete, open a command prompt and type `nvidia-ctk --version` . This command will display the installed CUDA version if the installation was successful.

##### Linux

**Check Prerequisites:** Verify you have a CUDA-capable GPU, a supported version of Linux (like Ubuntu, Fedora, or RHEL), and the gcc compiler installed. You can check for a CUDA-capable GPU by running lspci | grep -i nvidia in the terminal.

**Download the CUDA Toolkit:** Visit the NVIDIA CUDA downloads page and select your operating system, architecture, and distribution. You'll be given a choice between a distribution-specific package (e.g., .deb for Debian/Ubuntu, or .rpm for Fedora/RHEL) or a distribution-independent runfile. The distribution-specific packages are often recommended.

**Install the Toolkit:** Follow the instructions provided on the download page for your selected installer type. This typically involves adding the NVIDIA GPG key, adding the CUDA repository, and then running a command like sudo apt install cuda (for Ubuntu) or sudo yum install cuda (for RHEL/Fedora).

**Configure Environment Variables:** After installation, you must update your PATH and LD_LIBRARY_PATH environment variables. You can add lines like export PATH=/usr/local/cuda-<version>/bin${PATH:+:${PATH}} to your shell's configuration file (e.g., ~/.bashrc).

**Verify Installation:** Run `nvidia-ctk --version` in your terminal to check the installation.

##### macOS

**Note:** NVIDIA has deprecated the use of the CUDA Toolkit for development on macOS. While there are no longer tools that use macOS as a target environment, you can download macOS host versions of some tools (like Nsight Systems or Nsight Compute) to launch profiling and debugging sessions on supported target platforms. This means you can use your Mac as a host machine to develop for a different system with a supported GPU, but you can no longer run CUDA applications directly on a macOS system.

If you have a very old Mac with a supported NVIDIA GPU, you might be able to find and install older versions of the CUDA Toolkit. The process for these older versions typically involved:

**Checking System Requirements:** Verify you have a supported macOS version and a compatible NVIDIA GPU.

**Downloading the Installer:** Download the .pkg installer from the NVIDIA website.

**Installing:** Run the installer and follow the prompts.

**Setting Up Environment Variables:** Manually set the PATH and DYLD_LIBRARY_PATH variables in your shell configuration.

**Verifying:** Use nvcc -V in the terminal to confirm the installation.

#### 2. Run Docker

We would run the docker container which trigger `main.py` as an entery point.

```bash
docker run -v $PWD:/app --rm --gpus all compu-lstm 
```

NOTE: If you're going to run with python environment, make sure to uncomment tensorflow installation in the `requirements.txt` file


### 3. Create a virtual environment and activate it

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

Make sure to uncomment `'tensorflow[and-cuda]'==2.15.1` line in the `requirements.txt` file

```bash
pip install -r requirements.txt
```


---

## 🛠️ Step-by-Step Guide

The entery point for processing, training and preducing result is `main.py` file.

### Step [1]:

Check `src/configs.py` file for training, preprocessing and melody configurations.

### Step [2]:

Check `src/preprocess.py` file for preprocessing data code.

### Step [3]:

Check `src/train.py` file for compiling, build and training code.

### Step [4]:

Check `src/melodygenerator.py` file in which loads our trained model and uses it to predict the next melody until it finish.

### Step [5]:

Check `main.py` for altering the initial musical sequence which the model uses as initial input to predict the next sequences of music.

## 📊 Results

The final result is midi file in path `outputs/mel.mid`.

In order to listen to the generated music, visit https://midiplayer.ehubsoft.net/ and upload your file there.

### For futher improvement

Experiment with adjusting the initial sequence to observe how the model generates different melodies

You can also explore the impact of training the model for a longer duration or modifying the learning rate
