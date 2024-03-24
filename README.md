# CPEN455 2023 W2 Course Project: Conditional PixelCNN++ for Image Classification

This repo is for CPEN 455 course project 2023 Winter Term 2 at UBC. **The goal of this project is to implement the Conditional PixelCNN++ model and train it on the given dataset.** After that, the model can both generate new images and classify the given images. **we would evaluate the model based on both the generation performance and classification performance.**



## Project Guidelines

PixelCNN++ is a powerful generative model with tractable likelihood. It models the joint distribution of pixels over an image x as the following product of conditional distributions.

<img src="https://cdn-uploads.huggingface.co/production/uploads/65000e786a230e55a65c45ad/-jZg8HEMyFnpduNsi-Alt.png" width = "500" align="center"/>

where x_i is a single pixel.

Given a class embedding c, PixelCNN++ can be extended to conditional generative tasks following:

<img src="https://cdn-uploads.huggingface.co/production/uploads/65000e786a230e55a65c45ad/_jv7O2Z_1s1oYLXjIqS1V.png" width = "260" align="center"/>

In this case, with a trained conditional PixelCNN++, we could directly apply it to the zero-shot image classification task by:

<img src="https://cdn-uploads.huggingface.co/production/uploads/65000e786a230e55a65c45ad/DkKS0zI5FZ6wpw0WZtdSD.png" width = "260" align="center"/>

**Task:** For our final project, you are required to achieve the following tasks
* We will provide you with codes for an unconditional PixelCNN++. You need to adapt it to conditional image generation task and train it on our provided database.
  
* Complete `generation_evaluation.py` to conditionally generate images and evaluate the generated images using FID score.
  * Modify the sample function to get the generated images from the model and ensure the generated images are saved to the gen_data_dir(line 12-18)
  * Modify how you call your sample function(line 31)

* Complete a classification function in `classification_evaluation.py` to convert the output of conditional PixelCNN++ to the prediction labels when given a new image.
  * Replace the random classifier with your trained model.(line 64-68)
  * modify the get_label function to get the predicted label.(line 18-24)
* Please DO NOT change any definitions in the two interface classes `generation_evaluation.py` and `classification_evaluation.py`.

  

## Basic tools
We TAs recommend several tools which will help you debug and monitor the training process:

1.wandb: wandb is a tool that helps you monitor the training process. You can see the loss, accuracy, and other metrics in real time. You can also see the generated images and the model structure. You can find how to use wandb in the following link: https://docs.wandb.ai/quickstart

2.tensorboard: tensorboard is another tool that helps you monitor the training process. You can find how to use tensorboard in the following link: https://www.tensorflow.org/tensorboard/get_started

3.pdb: pdb is a python debugger. You can use it to debug your code. You can find how to use pdb in the following link: https://docs.python.org/3/library/pdb.html

4.conda: conda is a package manager. You can use it to create a virtual environment and install the required packages. You can find how to use conda in the following link: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


## Original PixelCNN++ code
we provided the code for the PixelCNN++ model. Before you run the code, you need to install the required packages by running the following command:
```
conda create -n cpen455 python=3.10.13
conda activate cpen455
conda install pip3
```
when you type the command `which pip3`, you should see the path of the pip3 in the virtual environment but not in the system pip3 path.

if you make sure the pip3 is in the virtual environment, you can install pytorch via this touorial: https://pytorch.org/get-started/locally/, you should choose the right version of command according to your system, for example, if you use linux with cuda support, you should use the following command:
```
pip3 install torch torchvision torchaudio
```
After you install the pytorch, you can install the other required packages by running the following command:
```
pip install -r requirements.txt
```

Please note that we guarantee that the requirements.txt file includes all the Python packages necessary to complete the final project. Therefore, **please DO NOT install any third-party packages.** If this results in the inability to run the submitted code later on, you may need to take responsibility. If you have any questions regarding Python packages, please contact the teaching assistant.

And then, you can run the code by running the following command:
```
python pcnn_train.py \
--batch_size 16 \
--sample_batch_size 16 \
--sampling_interval 25 \
--save_interval 25 \
--dataset cpen455 \
--nr_resnet 1 \
--lr_decay 0.999995 \
--max_epochs 500 \
--en_wandb True \
```

If you want to go into more detail about Pixelcnn++, you can find the original paper at the following link: https://arxiv.org/abs/1701.05517

And there are some repositories that implement the PixelCNN++ model. You can find them in the following link:

1. Original PixelCNN++ repository implemented by OpenAI: https://github.com/openai/pixel-cnn

2. Pytorch implementation of PixelCNN++: https://github.com/pclucas14/pixel-cnn-pp

# Dataset

In our provided code base, we have included the data required to train conditional PixelCNN++. The directory structure is as follows:
```
data
├── test
├── train
└── validation
```

Among them, the `train` directory contains 4160 labeled training images, divided into 4 different categories. The `validation` directory contains 520 labeled validation images. The `test` directory contains 520 unlabeled testing images used for evaluating model performance.


The ground truth labels for the training set and validation set are stored in the `data/train.csv` and `data/validation.csv` respectively. These two `.csv` files contain two columns: `id` and `label`, as shown below:

```
id, label
0000000.jpg,1
0000001.jpg,0
0000002.jpg,3
0000003.jpg,1
0000005.jpg,3
0000006.jpg,3
0000007.jpg,2
0000008.jpg,0
```

## Evaluation

For the evaluation of model performance, we assess the quality of images generated by conditional PixelCNN++ and the accuracy of classification separately. 

For classification accuracy, we evaluate using both **accuracy** and **F1 score**. You can submit your classification results through the [project Hugging Face challenge page](https://huggingface.co/spaces/CPEN455-23W2/CPEN45523W2CourseProject). Please see the Submission Guidelines tab. Our system will calculate accuracy and F1 score based on your submission and then update the leaderboard accordingly.

For assessing the quality of generated images, we provided an evaluation interface function in `generation_evaluation.py` using the **FID score** to gauge the quality. After the final project deadline, we will run all submitted code on our system and execute the FID evaluation function. It is essential to ensure that your code runs correctly and can reproduce the evaluation results reported in the project. Failure to do so may result in corresponding deductions.

Evaluation of model performance will affect a portion of the final score, but not all of it. After deadlines, we will attempt to reproduce all submitted code, and any cheating discovered will result in deductions and appropriate actions taken. The quality of the code, the completeness of the project, and the ability to reproduce results will all be considered in determining the final score.

## Academic Integrity Guidelines for the Course Project:

In developing your model, you are permitted to utilize any functions available in PyTorch and to consult external resources. However, it is imperative to properly acknowledge all sources and prior work utilized.

Violations of academic integrity will result in a grade of ZERO. These violations include:

1. Extensive reuse or adaptation of existing methods or code bases without proper citation in both your report and code.
2. Use of tools like ChatGPT or Copilot for code generation without proper acknowledgment, including details of prompt-response interactions.
3. Deliberate attempts to manipulate the testing dataset in order to extract ground-truth labels or employ discriminative classifiers.
4. Intentional submission of fabricated results to the competition leaderboard.
5. Any form of academic dishonesty, such as sharing code, model checkpoints, or inference results with other students.

Additionally, we request the following:

1. Retain both the logits and the final discrete labels from your classification results. While logits are not used for evaluation, they may be requested during investigations of academic integrity.
2. Refrain from distributing any outputs generated by your model, including, but not limited to, images or inferred probabilities from provided datasets.

Adhering to these guidelines is crucial for maintaining a fair and honest academic environment. Thank you for your cooperation.
