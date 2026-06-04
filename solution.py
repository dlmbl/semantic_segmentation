# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: 03-segmentation
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Segmentation
#
# <hr style="height:2px;">
#
# <div class="alert alert-info">
#
# This exercise has two parts:
# - **Semantic Segmentation**, where we adapt our 2D U-Net for better nuclei segmentations of the Kaggle Nuclei dataset.
# - **Instance Segmentation**, where we adapt our 2D U-Net for instance segmentations of cells in the TissueNet dataset.
# </div>
#
# Written by William Patton, Vijay Venu, Valentyna Zinchenko, Arlo Sheridan, Larissa Heinrich, Carsen Stringer, and Constantin Pape.

# %% [markdown]
# <div class="alert alert-block alert-danger">
# <b>Conda Kernel</b>: Please use the kernel <code style="color: black">03-segmentation</code> for this exercise
# </div>

# %% [markdown]
# <hr style="height:2px;">
#
# ## The libraries

# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import subprocess
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import torchvision.transforms.v2 as transforms_v2

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %%
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()


# %% [markdown]
# # Part I: Semantic Segmentation
#
# <div class="alert alert-info">
#
# **Overview:**
#
# 1. Prepare the 2D U-Net baseline model and validation dataset.
# 2. Implement and use the Dice coefficient as an evaluation metric for the baseline model.
# 3. Improve metrics by experimenting with:
#     - Data augmentations
#     - Loss functions
#     - (bonus) Group Normalization, U-Net architecture
# </div>
#
# ## Section 0: What we have so far
#
# You have already implemented a U-Net architecture in the previous exercise. We will use it as a starting point for this exercise.
# You should also already have the dataset and the dataloader implemented, along with a simple train loop with MSELoss.
# Let us go ahead and visualize some of the data along with some predictions to see how we are doing.

# %% [markdown]
# Our goal is to produce a model that can take an image as input and produce a segmentation as shown in this table.
#
# | Image | Mask | Prediction |
# | :-: | :-: | :-: |
# | ![image](static/img_0.png) | ![mask](static/mask_0.png) | ![pred](static/pred_0.png) |
# | ![image](static/img_1.png) | ![mask](static/mask_1.png) | ![pred](static/pred_1.png) |

# %%
from local import (
    NucleiDataset,
    show_random_dataset_image,
    show_random_dataset_image_with_prediction,
    show_random_augmentation_comparison,
    train,
)
from dlmbl_unet import UNet


# %% [markdown]
#
# *Note*: We are artificially making our validation data worse. This dataset
# was chosen to be reasonable to segment in the amount of time it takes to
# run this exercise. However this means that some techniques like augmentations
# aren't as useful as they would be on a more complex dataset. So we are
# artificially adding noise to the validation data to make it more challenging.


# %%
def salt_and_pepper_noise(image, amount=0.05):
    """
    Add salt and pepper noise to an image
    """
    out = image.clone()
    num_salt = int(amount * image.numel() * 0.5)
    num_pepper = int(amount * image.numel() * 0.5)

    # Add Salt noise
    coords = [
        torch.randint(0, i - 1, [num_salt]) if i > 1 else [0] * num_salt
        for i in image.shape
    ]
    out[tuple(coords)] = 1

    # Add Pepper noise
    coords = [
        torch.randint(0, i - 1, [num_pepper]) if i > 1 else [0] * num_pepper
        for i in image.shape
    ]
    out[tuple(coords)] = 0

    return out


# %%
train_data = NucleiDataset("nuclei_train_data", transforms_v2.RandomCrop(256))
train_loader = DataLoader(train_data, batch_size=5, shuffle=True, num_workers=8)
val_data = NucleiDataset(
    "nuclei_val_data",
    transforms_v2.RandomCrop(256),
    img_transform=transforms_v2.Lambda(salt_and_pepper_noise),
)
val_loader = DataLoader(val_data, batch_size=5)

# %%
unet = UNet(depth=4, in_channels=1, out_channels=1, num_fmaps=2).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters())

for epoch in range(10):
    train(unet, train_loader, optimizer, loss, epoch, device=device)

# %%
# Show some predictions on the train data
show_random_dataset_image_with_prediction(train_data, unet, device)


# %%
# Show some predictions on the validation data
show_random_dataset_image_with_prediction(val_data, unet, device)
# %% [markdown]
#
# <div class="alert alert-block alert-info">
#     <p><b>Task 0.1</b>: Are the predictions good enough? Take some time to try to think about
#     what could be improved and how that could be addressed. If you have time try training a second
#     model and see which one is better</p>
# </div>

# %% [markdown] tags=["task"]
# Write your answers here:
# <ol>
#     <li></li>
#     <li></li>
#     <li></li>
# </ol>

# %% [markdown] tags=["solution"]
# Write your answers here:
# <ol>
#     <li> Evaluation metric for better understanding of model performance so we can compare. </li>
#     <li> Augments for generalization to validaiton. </li>
#     <li> Loss function for better performance on lower prevalence classes. </li>
# </ol>

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 0 </h2>
# <p>We will go over the steps up to this point soon. By this point you should have imported and re-used
# code from previous exercises to train a basic UNet.</p>
# <p>The rest of this exercise will focus on tailoring our network to semantic segmentation to improve
# performance. The main areas we will tackle are:</p>
# <ol>
#   <li> Evaluation
#   <li> Augmentation
#   <li> Activations/Loss Functions
# </ol>
#
# </div>

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 1: Evaluation

# %% [markdown]
# One of the most important parts of training a model is evaluating it. We need to know how well our model is doing and if it is improving.
# We will start by implementing a metric to evaluate our model. Evaluation is always specific to the task, in this case semantic segmentation.
# We will use the [Dice Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) to evaluate the network predictions.
# We can use it for validation if we interpret set $a$ as predictions and $b$ as labels. It is often used to evaluate segmentations with sparse
# foreground, because the denominator normalizes by the number of foreground pixels.
# The Dice Coefficient is closely related to Jaccard Index / Intersection over Union.
# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.1</b>: Fill in implementation details for the Dice Coefficient. You can do this without using any torch or numpy functions.
# </div>


# %% tags=["task"]
# Sorensen-Dice coefficient implemented in torch.
# Takes prediction and target tensors of the same shape with values in {0, 1} and returns a single score in [0, 1],
# where 0 is the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # 2 * sum(prediction * target) / (sum(prediction**2) + sum(target**2))
    def forward(self, prediction, target):
        numerator = ... # TODO
        denominator = ... # TODO
        return numerator / denominator.clamp(min=self.eps)


# %% tags=["solution"]
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        numerator = 2 * (prediction * target).sum()
        denominator = (prediction ** 2).sum() + (target ** 2).sum()
        return numerator / denominator.clamp(min=self.eps)


# %% [markdown]
# <div class="alert alert-block alert-warning">
#     Test your Dice Coefficient here, are you getting the right scores?
# </div>

# %%
dice = DiceCoefficient()
target = torch.tensor([0.0, 1.0])
good_prediction = torch.tensor([0.0, 1.0])
bad_prediction = torch.tensor([0.0, 0.0])
wrong_prediction = torch.tensor([1.0, 0.0])

assert dice(good_prediction, target) == 1.0, dice(good_prediction, target)
assert dice(bad_prediction, target) == 0.0, dice(bad_prediction, target)
assert dice(wrong_prediction, target) == 0.0, dice(wrong_prediction, target)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.2</b>: What happens if your predictions are not discrete elements of {0,1}?
#     <ol>
#         <li>What happens to the Dice score if the predictions are in range (0,1)?</li>
#         <li>What happens to the Dice score if the predictions are in range (-∞, ∞)?</li>
#     </ol>
# </div>

# %% [markdown] tags=["task"]
# Answer:
# 1) ...
#
# 2) ...

# %% [markdown] tags=["solution"]
# Answer:
# 1) Score remains between (0,1) with 0 being the worst score and 1 being the best. This case
# essentially gives you the Dice Loss and can be a good alternative to cross entropy.
#
# 2) Scores will fall in the range of [-1,1]. Overly confident scores will be penalized i.e.
# if the target is [0,1] then a prediction of [0,2] will score higher than a prediction of [0,3].

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 1 </h2>
#
# This is a good place to stop for a moment. If you have extra time look into some extra
# evaluation functions or try to implement your own without hints.
# Some popular alternatives to the Dice Coefficient are the Jaccard Index and Balanced F1 Scores.
# You may even have time to compute the evaluation score between some of your training and
# validation predictions to their ground truth using our previous models.
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 1.3</b>: Fix in all the TODOs (3) to make the validate function work. If confused, you can use this
# <a href="https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html">PyTorch tutorial</a> as a template
# </div>


# %% tags=["task"]
# run validation after training epoch
def validate(
    model,
    loader,
    loss_function,
    metric,
    step=None,
    tb_logger=None,
    device=None,
):
    """
    Evaluate model performance on validation data.

    Args:
        model: PyTorch model to validate
        loader: DataLoader containing validation data
        loss_function: Loss function to compute validation loss between prediction and target.
        metric: Metric function to evaluate segmentation against ground-truth labels.
        step: Current training step for logging (required if tb_logger provided)
        tb_logger: TensorBoard logger for recording metrics (optional)
        device: Torch device to run validation on (optional)

    Returns:
        None
    """

    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # TODO: evaluate this example with the given loss and metric
            prediction = ... # TODO
            # We *usually* want the target to be the same type as the prediction
            # however this is very dependent on your choice of loss function and
            # metric. If you get errors such as "RuntimeError: Found dtype Float but expected Short"
            # then this is where you should look.
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
            val_loss += ... # TODO
            val_metric += ... # TODO

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)

    if tb_logger is not None:
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(
            tag="val_metric", scalar_value=val_metric, global_step=step
        )
        # we always log the last validation images
        tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=step)
        tb_logger.add_images(tag="val_target", img_tensor=y.to("cpu"), global_step=step)
        tb_logger.add_images(
            tag="val_prediction", img_tensor=prediction.to("cpu"), global_step=step
        )

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
            val_loss, val_metric
        )
    )


# %% tags=["solution"]
# run validation after training epoch
def validate(
    model,
    loader,
    loss_function,
    metric,
    step=None,
    tb_logger=None,
    device=None,
):
    """
    Evaluate model performance on validation data.

    Args:
        model: PyTorch model to validate
        loader: DataLoader containing validation data
        loss_function: Loss function to compute validation loss between prediction and target.
        metric: Metric function to evaluate segmentation against ground-truth labels.
        step: Current training step for logging (required if tb_logger provided)
        tb_logger: TensorBoard logger for recording metrics (optional)
        device: Torch device to run validation on (optional)

    Returns:
        None
    """

    if device is None:
        # You can pass in a device or we will default to using
        # the gpu. Feel free to try training on the cpu to see
        # what sort of performance difference there is
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # set model to eval mode
    model.eval()
    model.to(device)

    # running loss and metric values
    val_loss = 0
    val_metric = 0

    # disable gradients during validation
    with torch.no_grad():
        # iterate over validation loader and update loss and metric values
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)
            # We *usually* want the target to be the same type as the prediction
            # however this is very dependent on your choice of loss function and
            # metric. If you get errors such as "RuntimeError: Found dtype Float but expected Short"
            # then this is where you should look.
            if y.dtype != prediction.dtype:
                y = y.type(prediction.dtype)
            val_loss += loss_function(prediction, y).item()
            val_metric += metric(prediction > 0.5, y).item()

    # normalize loss and metric
    val_loss /= len(loader)
    val_metric /= len(loader)

    if tb_logger is not None:
        assert (
            step is not None
        ), "Need to know the current step to log validation results"
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)
        tb_logger.add_scalar(
            tag="val_metric", scalar_value=val_metric, global_step=step
        )
        # we always log the last validation images
        tb_logger.add_images(tag="val_input", img_tensor=x.to("cpu"), global_step=step)
        tb_logger.add_images(tag="val_target", img_tensor=y.to("cpu"), global_step=step)
        tb_logger.add_images(
            tag="val_prediction", img_tensor=prediction.to("cpu"), global_step=step
        )

    print(
        "\nValidate: Average loss: {:.4f}, Average Metric: {:.4f}\n".format(
            val_loss, val_metric
        )
    )


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 1.4</b>: Evaluate your first model using the Dice Coefficient. How does it perform? If you trained two models,
#     do the scores agree with your visual determination of which model was better?
# </div>

# %% tags=["task"]

# Evaluate your model here
validate(...)

# %% tags=["solution"]

# Evaluate your model here

validate(
    unet,
    val_loader,
    loss_function=torch.nn.MSELoss(),
    metric=DiceCoefficient(),
    step=0,
    device=device,
)

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 2</h2>
#
# We have finished writing the evaluation function. We will go over the code up to this point soon.
# Next we will work on augmentations to improve the generalization of our model.
#
# </div>
#
# <hr style="height:2px;">

# %% [markdown]
# ## Section 2: Augmentation
# Often our models will perform better on the evaluation dataset if we augment our training data.
# This is because the model will be exposed to a wider variety of data that will hopefully help
# cover the full distribution of data in the validation set. We will use the <code style="color: black">torchvision.transforms.v2</code> modules
# to augment our data.


# %% [markdown]
# PS: PyTorch already has quite a few possible data transforms, so if you need one, check
# [here](https://pytorch.org/vision/stable/transforms.html#transforms-on-pil-image-and-torch-tensor).
# The biggest problem with them is that they are clearly separated into transforms applied to PIL
# images (remember, we initially load the images as PIL.Image?) and torch.tensors (remember, we
# converted the images into tensors by calling transforms.ToTensor()?). This can be incredibly
# annoying if for some reason you might need to transorm your images to tensors before applying any
# other transforms or you don't want to use PIL library at all.

# %% [markdown]
# Here is an example augmented dataset. Use it to see how it affects your data, then play around with at least
# 2 other augmentations.
# There are two types of augmentations: <code style="color: black">transform</code> and <code style="color: black">img_transform</code>. The first one is applied to both the
# image and the mask, the second is only applied to the image. This is useful if you want to apply augmentations
# that spatially distort your data and you want to make sure the same distortion is applied to the mask and image.
# <code style="color: black">img_transform</code> is useful for augmentations that don't make sense to apply to the mask, like blurring.

# %%
train_data = NucleiDataset("nuclei_train_data", transforms_v2.RandomCrop(256))

# Note this augmented data uses extreme augmentations for visualization. It will not train well
example_augmented_data = NucleiDataset(
    root_dir="nuclei_train_data",
    transform=transforms_v2.Compose(
        [transforms_v2.RandomRotation(45), transforms_v2.RandomCrop(256)]
    ),
    img_transform=transforms_v2.Compose([transforms_v2.GaussianBlur(21, sigma=10.0)]),
)

# %%
show_random_augmentation_comparison(train_data, example_augmented_data)

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.1</b>: Now create an augmented dataset with an augmentation of your choice.
#       **hint**: Using the same augmentation as was applied to the validation data will
#      likely be optimal. Bonus points if you can get good results without the custom noise.
# </div>

# %% tags=["task"]
augmented_data = ...

# %% tags=["solution"]
augmented_data = NucleiDataset(
    "nuclei_train_data",
    transforms_v2.Compose(
        [transforms_v2.RandomRotation(45), transforms_v2.RandomCrop(256)]
    ),
    img_transform=transforms_v2.Compose([transforms_v2.Lambda(salt_and_pepper_noise)]),
)


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.2</b>: Now retrain your model with your favorite augmented dataset. Did your model improve?
# </div>


# %% tags=["task"]

unet = UNet(depth=4, in_channels=1, out_channels=1, num_fmaps=2).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters())
augmented_loader = DataLoader(augmented_data, batch_size=5, shuffle=True, num_workers=8)

# TODO: train your model on the augmented dataset
...

# %% tags=["solution"]

unet = UNet(depth=4, in_channels=1, out_channels=1, num_fmaps=2).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters())
augmented_loader = DataLoader(augmented_data, batch_size=5, shuffle=True, num_workers=8)

for epoch in range(10):
    train(unet, augmented_loader, optimizer, loss, epoch, device=device)

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 2.3</b>: Now evaluate your model. Did your model improve?
# </div>

# %% tag=["task"]
validate(...)

# %% tags=["solution"]
validate(unet, val_loader, loss, DiceCoefficient(), device=device)

# %% [markdown] tags=["solution"]
# <hr style="height:2px;">

# %% [markdown]
# ## Section 3: Loss Functions

# %% [markdown]
# The next step to do would be to improve our loss function - the metric that tells us how
# close we are to the desired output. This metric should be differentiable, since this
# is the value to be backpropagated. The are
# [multiple losses](https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html)
# we could use for the segmentation task.
#
# Take a moment to think which one is better to use. If you are not sure, don't forget
# that you can always google! Before you start implementing the loss yourself, take a look
# at the [losses](https://pytorch.org/docs/stable/nn.html#loss-functions) already implemented
# in PyTorch. You can also look for implementations on GitHub.

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task 3.1</b>: Implement your loss (or take one from pytorch):
# </div>

# %% tags=["task"]
# implement your loss here or initialize the one of your choice from PyTorch
loss_function: torch.nn.Module = ...

# %% tags=["solution"]
# implement your loss here or initialize the one of your choice from PyTorch
loss_function: torch.nn.Module = nn.BCELoss()

# %% [markdown]
# <div class="alert alert-block alert-warning">
#     Test your loss function here, is it behaving as you'd expect?
# </div>

# %%
target = torch.tensor([0.0, 1.0])
good_prediction = torch.tensor([0.01, 0.99])
bad_prediction = torch.tensor([0.4, 0.6])
wrong_prediction = torch.tensor([0.9, 0.1])

good_loss = loss_function(good_prediction, target)
bad_loss = loss_function(bad_prediction, target)
wrong_loss = loss_function(wrong_prediction, target)

assert good_loss < bad_loss
assert bad_loss < wrong_loss

# Can your loss function handle predictions outside of (0, 1)?
# Some loss functions will be perfectly happy with this which may
# make them easier to work with, but predictions outside the expected
# range will not work well with our soon to be discussed evaluation metric.
out_of_bounds_prediction = torch.tensor([-0.1, 1.1])

try:
    oob_loss = loss_function(out_of_bounds_prediction, target)
    print("Your loss supports out-of-bounds predictions.")
except RuntimeError as e:
    print(e)
    print("Your loss does not support out-of-bounds predictions")

# %% [markdown]
# Pay close attention to whether your loss function can handle predictions outside of the range (0, 1).
# If it can't, theres a good chance that the activation function requires a specific activation before
# being passed into the loss function. This is a common source of bugs in DL models. For example, trying
# to use the <code style="color: black">torch.nn.BCEWithLogitsLoss</code> loss function with a model that has a sigmoid activation will
# result in abysmal performance, wheras using the <code style="color: black">torch.nn.BCELoss</code> loss function with a model that has
# no activation function will likely error out and fail to train.

# %%
# Now lets start experimenting. Start a tensorboard logger to keep track of experiments.
# start a tensorboard writer
logger = SummaryWriter("runs/Unet")


# Function to find an available port and launch TensorBoard on the browser
def launch_tensorboard(log_dir):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    tensorboard_cmd = f"tensorboard --logdir={log_dir} --port={port}"
    process = subprocess.Popen(tensorboard_cmd, shell=True)
    print(
        f"TensorBoard started at http://localhost:{port}. \n"
        "If you are using VSCode remote session, forward the port using the PORTS tab next to TERMINAL."
    )
    return process


launch_tensorboard("runs")


# %%
model = UNet(
    depth=4,
    in_channels=1,
    out_channels=1,
    num_fmaps=2,
    final_activation=torch.nn.Sigmoid(),
).to(device)

# use adam optimizer
optimizer = torch.optim.Adam(model.parameters())

# build the dice coefficient metric
metric = DiceCoefficient()

# train for $25$ epochs
# during the training you can inspect the
# predictions in the tensorboard
n_epochs = 25
for epoch in range(n_epochs):
    # train
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
        log_interval=25,
        tb_logger=logger,
        device=device,
    )
    step = epoch * len(train_loader)
    # validate
    validate(model, val_loader, loss_function, metric, step=step, tb_logger=logger)


# %% [markdown]
# Check the value of your validation metric at end of the training. Even if the number seems good enough,
# an equally important thing to check is: Open the Images tab in your Tensorboard and compare
# predictions to targets. Do your predictions look reasonable? Are there any obvious failure cases?
# If nothing is clearly wrong, let's see if we can still improve the model performance by changing
# the model or the loss
#


# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h2>Checkpoint 3</h2>
#
# This is the end of Part I (semantic segmentation). We will go over all of the code up until this point shortly.
# While you wait you are encouraged to try alternative loss functions, evaluation metrics, augmentations,
# and networks. After this come additional exercises if you are interested and have the time.
#
# </div>
# <hr style="height:2px;">

# %% [markdown]
# ## Additional Exercises
#
# 1. Modify and evaluate the following architecture variants of the U-Net:
#     * use [GroupNorm](https://pytorch.org/docs/stable/nn.html#torch.nn.GroupNorm) to normalize convolutional group inputs
#     * use more layers in your U-Net.
#
# 2. Use the Dice Coefficient as loss function. Before we only used it for validation, but it is differentiable
# and can thus also be used as loss. Compare to the results from exercise 2.
# Hint: The optimizer we use finds minima of the loss, but the minimal value for the Dice coefficient corresponds
# to a bad segmentation. How do we need to change the Dice Coefficient to use it as loss nonetheless?
#
# 3. Compare the results of these trainings to the first one. If any of the modifications you've implemented show
# better results, combine them (e.g. add both GroupNorm and one more layer) and run trainings again.
# What is the best result you could get?

# %% [markdown]
#
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.1</b>: Modify the <code>ConvBlockGN</code> class in <code>bonus_unet.py</code> to include GroupNorm layers (use <code>num_groups=2</code>). Then in <code>UNetGN</code>, replace the two <code>ConvBlock</code> calls with <code>ConvBlockGN</code>.
# </div>


# %%
# See the original U-Net for an example of how to build the convolutional block
# We want operation -> activation -> normalization (2x)
# Hint: Group norm takes a "num_groups" argument. Use 2 to match the solution
# Task: Modify the bonus_unet.py file as needed and save the changes before you run this cell

from bonus_unet import UNetGN

# %% tags=["solution"]
"""
Changes to make to the ConvBlockGN class in bonus_unet.py:

        self.conv_pass = torch.nn.Sequential(
            ...
        )
    becomes:
        self.conv_pass = torch.nn.Sequential(
            convops[ndim](
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(2, out_channels),
            convops[ndim](
                out_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(2, out_channels),
        )


Changes to make to the UNetGN class in bonus_unet.py:

    lines 231 and 241: change `ConvBlock` to `ConvBlockGN`

"""

from bonus_unet import UNetGN

# %%
model = UNetGN(
    depth=4,
    in_channels=1,
    out_channels=1,
    num_fmaps=2,
    final_activation=torch.nn.Sigmoid(),
).to(device)

optimizer = torch.optim.Adam(model.parameters())

metric = DiceCoefficient()

logger = SummaryWriter("runs/UNetGN")


# train for 40 epochs
# during the training you can inspect the
# predictions in the tensorboard
n_epochs = 40
for epoch in range(n_epochs):
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        model,
        val_loader,
        loss_function,
        metric,
        step=step,
        tb_logger=logger,
        device=device,
    )


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.2</b>: More Layers
# </div>

# %% tags=["task"]
# Experiment with more layers. For example UNet with depth 5

model = ...

optimizer = torch.optim.Adam(model.parameters())

metric = DiceCoefficient()

loss = torch.nn.BCELoss()

logger = SummaryWriter("runs/UNet5layers")

# %% tags=["solution"]
# Experiment with more layers. For example UNet with depth 5

model = UNet(
    depth=5,
    in_channels=1,
    out_channels=1,
    num_fmaps=2,
    final_activation=torch.nn.Sigmoid(),
).to(device)

optimizer = torch.optim.Adam(model.parameters())

metric = DiceCoefficient()

loss = torch.nn.BCELoss()

logger = SummaryWriter("runs/UNet5layers")

# %%
# train for 25 epochs
# during the training you can inspect the
# predictions in the tensorboard
n_epochs = 25
for epoch in range(n_epochs):
    train(
        model,
        train_loader,
        optimizer=optimizer,
        loss_function=loss,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        model, val_loader, loss, metric, step=step, tb_logger=logger, device=device
    )


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.3</b>: Dice Loss
#     Dice Loss is a simple inversion of the Dice Coefficient.
#     We already have a Dice Coefficient implementation, so now we just
#     need a layer that can invert it.
# </div>


# %% tags=["task"]
class DiceLoss(nn.Module):
    """ """

    def __init__(self, offset: float = 1):
        super().__init__()
        self.dice_coefficient = DiceCoefficient()

    def forward(self, x, y): ...


# %% tags=["solution"]
class DiceLoss(nn.Module):
    """
    This layer will simply compute the dice coefficient and then negate
    it with an optional offset.
    We support an optional offset because it is common to have 0 as
    the optimal loss. Since the optimal dice coefficient is 1, it is
    convenient to get 1 - dice_coefficient as our loss.

    You could leave off the offset and simply have -1 as your optimal loss.
    """

    def __init__(self, offset: float = 1):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(offset), requires_grad=False)
        self.dice_coefficient = DiceCoefficient()

    def forward(self, x, y):
        coefficient = self.dice_coefficient(x, y)
        return self.offset - coefficient


# %% tags=["task"]
# Now combine the Dice Coefficient layer with the Invert layer to make a Dice Loss
dice_loss = ...

# %% tags=["solution"]
# Now combine the Dice Coefficient layer with the Invert layer to make a Dice Loss
dice_loss = DiceLoss()

# %% tags=["task"]
# Experiment with Dice Loss
net = ...
optimizer = ...
metric = ...
loss_func = ...

# %% tags=["solution"]
# Experiment with Dice Loss
net = UNet(
    depth=4,
    in_channels=1,
    out_channels=1,
    num_fmaps=2,
    final_activation=torch.nn.Sigmoid(),
).to(device)
optimizer = torch.optim.Adam(net.parameters())
metric = DiceCoefficient()
loss_func = dice_loss

# %%
logger = SummaryWriter("runs/UNet_diceloss")

n_epochs = 40
for epoch in range(n_epochs):
    train(
        net,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_func,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        net, val_loader, loss_func, metric, step=step, tb_logger=logger, device=device
    )

# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.4</b>: Group Norm + Dice
# </div>


# %% tags=["task"]
net = ...
optimizer = ...
metric = ...
loss_func = ...

# %% tags=["solution"]
net = UNetGN(
    depth=4,
    in_channels=1,
    out_channels=1,
    num_fmaps=2,
    final_activation=torch.nn.Sigmoid(),
).to(device)
optimizer = torch.optim.Adam(net.parameters())
metric = DiceCoefficient()
loss_func = dice_loss

# %%
logger = SummaryWriter("runs/UNetGN_diceloss")

n_epochs = 40
for epoch in range(n_epochs):
    train(
        net,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_func,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        net, val_loader, loss_func, metric, step=step, tb_logger=logger, device=device
    )


# %% [markdown]
# <div class="alert alert-block alert-info">
#     <b>Task BONUS.5</b>: Group Norm + Dice + U-Net 5 Layers
# </div>

# %% tags=["task"]
net = ...
optimizer = ...
metric = ...
loss_func = ...

# %% tags=["solution"]
net = UNetGN(
    depth=5,
    in_channels=1,
    out_channels=1,
    num_fmaps=2,
    final_activation=torch.nn.Sigmoid(),
).to(device)
optimizer = torch.optim.Adam(net.parameters())
metric = DiceCoefficient()
loss_func = dice_loss

logger = SummaryWriter("runs/UNet5layersGN_diceloss")

n_epochs = 40
for epoch in range(n_epochs):
    train(
        net,
        train_loader,
        optimizer=optimizer,
        loss_function=loss_func,
        epoch=epoch,
        log_interval=5,
        tb_logger=logger,
        device=device,
    )
    step = epoch * len(train_loader)
    validate(
        net, val_loader, loss_func, metric, step=step, tb_logger=logger, device=device
    )

# %% [markdown]
# # Part II : Instance Segmentation :D
#
# So far, we were only interested in <code style="color: black">semantic</code> classes, e.g. foreground / background etc.
# But in many cases we not only want to know if a certain pixel belongs to a specific class, but also to which unique object (i.e. the task of "instance segmentation").
#
# For isolated objects, this is trivial, all connected foreground pixels form one instance, yet often instances are very close together or even overlapping. Thus we need to think a bit more how to formulate the targets / loss of our network.
#
# Furthermore, in instance segmentation the specific value of each label is arbitrary. Here we label each cell with a number and assign a color to each number giving us a segmentation mask. "Mask 1" and "Mask 2" are equivalently good segmentations even though the specific label of each cell is arbitrary.
#
# | Image | Mask 1| Mask 2|
# | :-: | :-: | :-: |
# | ![image](static/figure1/01_instance_image.png) | ![mask1](static/figure1/02_instance_teaser.png) | ![mask2](static/figure1/03_instance_teaser.png) |
#
# Once again: THE SPECIFIC VALUES OF THE LABELS ARE ARBITRARY
#
# This means that the model will not be able to learn, if tasked to predict the labels directly.
#
# Therefore we split the task of instance segmentation in two and introduce an intermediate target which must be:
#   1) learnable
#   2) post-processable into an instance segmentation
#
# In this exercise we will go over two common intermediate targets (signed distance transform and affinities),
# as well as the necessary pre and post-processing for obtaining the final segmentations.
#
# At the end of the exercise we will also compare to a pre-trained cellpose model.

# %% [markdown]
# ## Import Packages

# %%
from matplotlib.colors import ListedColormap
import numpy as np
import os
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt, map_coordinates
from local import train, plot_two, plot_three, plot_four
from tqdm import tqdm
import tifffile
import mwatershed as mws

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

# %%
NUM_THREADS = 8
NUM_EPOCHS = 80
# make sure gpu is available. Please call a TA if this cell fails
assert torch.cuda.is_available()

# %%
# Create a custom label color map for showing instances
np.random.seed(1)
colors = [[0, 0, 0]] + [list(np.random.choice(range(256), size=3)) for _ in range(254)]
label_cmap = ListedColormap(colors)


# %% [markdown]
# ## Section 1: Signed Distance Transform (SDT)
#
# First, we will use the signed distance transform as an intermediate learning objective.
#
# <i>What is the signed distance transform?</i>
# <br>  - The Signed Distance Transform indicates the distance from each specific pixel to the boundary of objects.
# <br>  - It is positive for pixels inside objects and negative for pixels outside objects (i.e. in the background).
# <br>  - Remember that deep learning models work best with normalized values, therefore it is important to scale the distance
#            transform. For simplicity, things are often scaled between -1 and 1.
# <br>  - As an example, here, you see the SDT (right) of the target mask (middle), below.

# %% [markdown]
# ![image](static/figure2/04_instance_sdt.png)
#

# %% [markdown]
# We will now add the code for preparing our data to learn the signed distance transform.


# %%
def compute_sdt(labels: np.ndarray, scale: int = 5):
    """Function to compute a signed distance transform."""
    dims = len(labels.shape)
    # Create a placeholder array of infinite distances
    distances = np.ones(labels.shape, dtype=np.float32) * np.inf
    for axis in range(dims):
        # Here we compute the boundaries by shifting the labels and comparing to the original labels
        # This can be visualized in 1D as:
        # a a a b b c c c
        #   a a a b b c c c
        #   1 1 0 1 0 1 1
        # Applying a half pixel shift makes the result more obvious:
        # a a a b b c c c
        #  1 1 0 1 0 1 1
        bounds = (
            labels[*[slice(None) if a != axis else slice(1, None) for a in range(dims)]]
            == labels[
                *[slice(None) if a != axis else slice(None, -1) for a in range(dims)]
            ]
        )
        # pad to account for the lost pixel
        bounds = np.pad(
            bounds,
            [(1, 1) if a == axis else (0, 0) for a in range(dims)],
            mode="constant",
            constant_values=1,
        )
        # compute distances on the boundary mask
        axis_distances = distance_transform_edt(bounds)

        # compute the coordinates of each original pixel relative to the boundary mask and distance transform.
        # Its just a half pixel shift in the axis we computed boundaries for.
        coordinates = np.meshgrid(
            *[
                (
                    range(axis_distances.shape[a])
                    if a != axis
                    else np.linspace(
                        0.5, axis_distances.shape[a] - 1.5, labels.shape[a]
                    )
                )
                for a in range(dims)
            ],
            indexing="ij",
        )
        coordinates = np.stack(coordinates)

        # Interpolate the distances to the original pixel coordinates
        sampled = map_coordinates(
            axis_distances,
            coordinates=coordinates,
            order=3,
        )

        # Update the distances with the minimum distance to a boundary in this axis
        distances = np.minimum(distances, sampled)

    # Normalize the distances to be between -1 and 1
    distances = np.tanh(distances / scale)

    # Invert the distances for pixels in the background
    distances[labels == 0] *= -1
    return distances


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.1</b>: Explain the <code style="color: black">compute_sdt</code> from the cell above.
# </div>

# %% [markdown]
# 1. _Why do we need to loop over dimensions? Couldn't we compute all at once?_
#
# 2. _What is the purpose of the pad?_
#
# 3. _What does meshgrid do?_
#
# 4. _Why do we use <code style="color: black">map_coordinates</code>?_
#
# 5. _bonus question: Is the pad sufficient to give us accurate distances at the edge of our image?_

# %% [markdown] tags=["solution"]
# 1. _Why do we need to loop over dimensions? Couldn't we compute all at once?_
# To get the distance to boundaries in each axis. Regardless of the shift we choose, we will always miss boundaries that line up perfectly with the offset. (shifting only by (1, 1) will miss diagonal boundaries).
#
# 2. _What is the purpose of the pad?_
# We lose a pixel when we compute the boundaries so we need to pad to cover the whole input image.
#
# 3. _What does meshgrid do?_
# It computes the index coordinate of every voxel. Offset by half on the dimension along which we computed boundaries because the boundaries sit half way between the voxels on either side of the boundary
#
# 4. _Why do we use <code style="color: black">map_coordinates</code>?_
# Boundaries are defined between pixels, not on individual pixels. So the distance from a pixel on a boundary to the boundary should be half of a pixel. Map Coordinates lets us get this interpolation
#
# 5. _bonus question: Is the pad sufficient to give us accurate distances at the edge of our image?_
# Kind of. If you assume this is the full image and no data exists outside the provided region, then yes. But if you have a larger image, then you cannot know the distance to the nearest out of view object. It might be visible given one more pixel, or there could never be another object.
# Depending on how you train, you may need to take this into account.

# %% [markdown]
# Below is a quick visualization of the signed distance transform (SDT).
# <br> Note that the output of the signed distance transform is not binary, a significant difference from semantic segmentation

# %%
# Visualize the signed distance transform using the function you wrote above.

root_dir = "tissuenet_data/train"  # the directory with all the training samples
samples = os.listdir(root_dir)
idx = np.random.randint(len(samples) // 3)  # take a random sample.
img = tifffile.imread(os.path.join(root_dir, f"img_{idx}.tif"))  # get the image
label = tifffile.imread(
    os.path.join(root_dir, f"img_{idx}_cyto_masks.tif")
)  # get the image
sdt = compute_sdt(label)
plot_three(img, label, sdt, label="SDT", label_cmap=label_cmap)


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.2</b>: Explain the scale parameter in <code style="color: black">compute_sdt</code>.
# </div>

# %% [markdown] tags=["task"]
# <b>Questions</b>:
# 1. _Why do we need to normalize the distances between -1 and 1?_
#
# 2. _What is the effect of changing the scale value? What do you think is a good default value?_
#

# %% [markdown] tags=["solution"]
# <b>Questions</b>:
# 1. _Why do we need to normalize the distances between -1 and 1?_
#   If the closest object to a pixel is outside the receptive field, the model cannot know whether the distance is 100 or 100_000. Squeezing large distances down to 1 or -1 makes the answer less ambiguous.
#
# 2. _What is the effect of changing the scale value? What do you think is a good default value?_
#   Increasing the scale is equivalent to having a wider boundary region. 5 seems reasonable.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.3</b>: <br>
#     Modify the <code style="color: black">SDTDataset</code> class below to produce the paired raw and SDT images.<br>
#   1. Fill in the <code style="color: black">create_sdt_target</code> method to return an SDT output rather than a label mask.<br>
#       - You may need to convert the input mask to a numpy array and use the helper function <code style="color: black">self.from_np</code> to convert back to a torch tensor.<br>
#       - Ensure that all final outputs are of torch tensor type, and are converted to float.<br>
#   2. Instantiate the dataset with a RandomCrop of size 128 and visualize the output to confirm that the SDT is correct.
# </div>


# %% tags=["task"]
class SDTDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks."""

    def __init__(self, root_dir, transform=None, img_transform=None, return_mask=False):
        self.root_dir = root_dir  # the directory with all the training samples
        self.num_samples = len(os.listdir(self.root_dir)) // 3  # list the samples
        self.return_mask = return_mask
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )
        self.from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))

        self.loaded_imgs = [None] * self.num_samples
        self.loaded_masks = [None] * self.num_samples
        for sample_ind in tqdm(range(self.num_samples)):
            img_path = os.path.join(self.root_dir, f"img_{sample_ind}.tif")
            image = self.from_np(tifffile.imread(img_path))
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(self.root_dir, f"img_{sample_ind}_cyto_masks.tif")
            mask = self.from_np(tifffile.imread(mask_path))
            self.loaded_masks[sample_ind] = mask

    # get the total number of samples
    def __len__(self):
        return self.num_samples

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # We'll be using the Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        # use the compute_sdt function to get the sdt
        sdt = self.create_sdt_target(mask)
        assert isinstance(sdt, torch.Tensor)
        assert sdt.dtype == torch.float32
        assert sdt.shape == mask.shape
        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.return_mask is True:
            return image, mask.unsqueeze(0), sdt.unsqueeze(0)
        else:
            return image, sdt.unsqueeze(0)

    def create_sdt_target(self, mask):

        ########## YOUR CODE HERE ##########

        target = ... # TODO
        
        return target


# %% tags=["solution"]
class SDTDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks."""

    def __init__(self, root_dir, transform=None, img_transform=None, return_mask=False):
        self.root_dir = root_dir  # the directory with all the training samples
        self.num_samples = len(os.listdir(self.root_dir)) // 3  # list the samples
        self.return_mask = return_mask
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms_v2.Compose(
            [
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )
        self.from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))

        self.loaded_imgs = [None] * self.num_samples
        self.loaded_masks = [None] * self.num_samples
        for sample_ind in tqdm(range(self.num_samples), desc="Reading Images"):
            img_path = os.path.join(self.root_dir, f"img_{sample_ind}.tif")
            image = self.from_np(tifffile.imread(img_path))
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(self.root_dir, f"img_{sample_ind}_cyto_masks.tif")
            mask = self.from_np(tifffile.imread(mask_path))
            self.loaded_masks[sample_ind] = mask

    # get the total number of samples
    def __len__(self):
        return self.num_samples

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # We'll be using the Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        sdt = self.create_sdt_target(mask)
        assert sdt.shape == mask.shape
        assert isinstance(sdt, torch.Tensor)
        assert sdt.dtype == torch.float32
        if self.img_transform is not None:
            image = self.img_transform(image)
        if self.return_mask is True:
            return image, mask.unsqueeze(0), sdt.unsqueeze(0)
        else:
            return image, sdt.unsqueeze(0)

    def create_sdt_target(self, mask):
        sdt_target_array = compute_sdt(mask.numpy())
        sdt_target = self.from_np(sdt_target_array)
        return sdt_target.float()


# %% tags=["task"]
# Create a dataset using a RandomCrop of size 128 (see torchvision.transforms.v2 imported as v2)
# documentation here: https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended
# Visualize the output to confirm your dataset is working.

train_data = SDTDataset("tissuenet_data/train", ...) # TODO
img, sdt = train_data[10]  # get the image and the distance transform
# We use the <code style="color: black">plot_two</code> function (imported in the first cell) to verify that our
# dataset solution is correct. The output should show 2 images: the raw image and
# the corresponding SDT.
plot_two(img, sdt[0], label="SDT")

# %% tags=["solution"]
# Create a dataset using a RandomCrop of size 128 (see torchvision.transforms.v2 imported as v2)
# documentation here: https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended
# Visualize the output to confirm your dataset is working.

train_data = SDTDataset("tissuenet_data/train", transforms_v2.RandomCrop(128))
img, sdt = train_data[10]  # get the image and the distance transform
# We use the plot_two function (imported in the first cell) to verify that our
# dataset solution is correct. The output should show 2 images: the raw image and
# the corresponding SDT.
plot_two(img, sdt[0], label="SDT")

# %% [markdown] tags=["task"]
# <div class="alert alert-block alert-info">
# <b>Task 1.4</b>: Understanding the Dataset.
# Our Dataset has some features that are not straightforward to understand or justify, and this is a good point
# to discuss them.
#
# 1. _What are we doing with the <code style="color: black">seed</code> variable in <code style="color: black">SDTDataset.__getitem__</code> and why? Can you predict what will go wrong when you delete the <code style="color: black">seed</code> code and rerun the previous cells visualization?_
#
#
# 2. _What is the purpose of the <code style="color: black">loaded_imgs</code> and <code style="color: black">loaded_masks</code> lists?_
#
# </div>

# %% [markdown] tags=["solution"]
# <div class="alert alert-block alert-info">
# <b>Task 1.4</b>: Understanding the Dataset.
# Our Dataset has some features that are not straightforward to understand or justify, and this is a good point
# to discuss them.
#
# 1. _What are we doing with the <code style="color: black">seed</code> variable in <code style="color: black">SDTDataset.__getitem__</code> and why? Can you predict what will go wrong when you delete the <code style="color: black">seed</code> code and rerun the previous cells visualization?_
# The seed variable is used to ensure that the same random transform is applied to the image and mask. If we don't use the seed, the image and mask will be transformed differently, leading to misaligned data.
#
# 2. _What is the purpose of the <code style="color: black">loaded_imgs</code> and <code style="color: black">loaded_masks</code> lists?_
# We load the images and masks into memory to avoid reading them from disk every time we access the dataset. This speeds up the training process. GPUs are very fast so
# we often need to put a lot of thought into how to provide data to them fast enough.
#
# </div>

# %% [markdown]
# Next, we will create our training dataset and data loader.
#

# %%
# TODO: You don't have to add extra augmentations, training will work without.
# But feel free to experiment here if you want to come back and try to get better results if you have time.
train_data = SDTDataset("tissuenet_data/train", transforms_v2.RandomCrop(128))
train_loader = DataLoader(
    train_data, batch_size=5, shuffle=True, num_workers=NUM_THREADS
)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 1.5</b>: Train the U-Net.
#
# In the cell below, fill in your code anywhere you see ...
#
# In this task, initialize the UNet, specify a loss function, learning rate, and optimizer, and train the model.<br>
# <br> For simplicity we will use a pre-made training function imported from <code style="color: black">local.py</code>. <br>
# <u>Hints</u>:<br>
#   - Loss function - [torch losses](https://pytorch.org/docs/stable/nn.html#loss-functions)
#   - Optimizer - [torch optimizers](https://pytorch.org/docs/stable/optim.html)
#   - Final Activation - there are a few options (only one is the best)
#       - [sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)
#       - [tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh)
#       - [relu](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)
# </div>

# %% tags=["task"]
# If you manage to get a loss close to 0.1, you are doing pretty well and can probably move on
unet = ...

learning_rate = ...
loss = ...
optimizer = ...

for epoch in range(NUM_EPOCHS):
    train(
        model=...,
        loader=...,
        optimizer=...,
        loss_function=...,
        epoch=...,
        log_interval=2,
        device=device,
    )

# %% tags=["solution"]
unet = UNet(
    depth=3,
    in_channels=2,
    out_channels=1,
    final_activation=torch.nn.Tanh(),
    num_fmaps=16,
    fmap_inc_factor=3,
    downsample_factor=2,
    padding="same",
)

learning_rate = 1e-4
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

for epoch in range(NUM_EPOCHS):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        log_interval=2,
        device=device,
    )

# %% [markdown]
# Now, let's apply our trained model and visualize some random samples. <br>
# First, we create a validation dataset. <br> Next, we sample a random image from the dataset and input into the model.

# %%
val_data = SDTDataset("tissuenet_data/test")
unet.eval()
idx = np.random.randint(len(val_data))  # take a random sample.
image, sdt = val_data[idx]  # get the image and the nuclei masks.
image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))
image = np.squeeze(image.cpu())
sdt = np.squeeze(sdt.cpu().numpy())
pred = np.squeeze(pred.cpu().detach().numpy())
plot_three(image, sdt, pred)

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 1</h2>
#
# At this point we have a model that does what we told it too, but do not yet have a segmentation. <br>
# In the next section, we will perform some post-processing and obtain segmentations from our predictions.

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 2: Post-Processing
# - See here for a nice overview: [open-cv-image watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html), although the specifics of our code will be slightly different
# - Given the distance transform (the output of our model), we first need to find the local maxima that will be used as seed points
# - The watershed algorithm then expands each seed out in a local "basin" until the segments touch or the boundary of the object is hit.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.1</b>: write a function to find the local maxima of the distance transform
#
# <u>Hint</u>: Look at the imports. <br>
# </div>

# %% tags=["task"]
from scipy.ndimage import label, maximum_filter


def find_local_maxima(distance_transform, min_dist_between_points):
    """
    Find the local maxima of the distance transform and generate seed points for our watershed.

    inputs:
        distance_transform: the distance transform of the image (2D numpy array)
        min_dist_between_points: the minimum distance between points (scalar)

    returns:
        seeds: the seeds for the watershed (2D numpy array with uniquely labelled seed points)
        number_of_seeds: the number of seeds (scalar)
    """

    # Apply a maximum filter to find the maximum value in each neighborhood
    # This creates an image where each pixel contains the maximum value from its neighborhood
    max_filtered = ... # TODO

    # Find where the original values equal the maximum filtered values
    # These are the local maxima - points that are >= all their neighbors
    maxima = ... # TODO

    # Uniquely label the local maxima
    seeds, number_of_seeds = ... # TODO

    return seeds, number_of_seeds


# %% tags=["solution"]
from scipy.ndimage import label, maximum_filter


def find_local_maxima(distance_transform, min_dist_between_points):
    """
    Find the local maxima of the distance transform and generate seed points for our watershed.

    inputs:
        distance_transform: the distance transform of the image (2D numpy array)
        min_dist_between_points: the minimum distance between points (scalar)

    returns:
        seeds: the seeds for the watershed (2D numpy array with uniquely labelled seed points)
        number_of_seeds: the number of seeds (scalar)
    """

    max_filtered = maximum_filter(distance_transform, min_dist_between_points)
    maxima = max_filtered == distance_transform
    # Uniquely label the local maxima
    seeds, number_of_seeds = label(maxima)

    return seeds, number_of_seeds


# %%
# test your function.
from local import test_maximum

test_maximum(find_local_maxima)

# %% [markdown]
# We now use this function to find the seeds for the watershed.

# %%
from skimage.segmentation import watershed


def watershed_from_boundary_distance(
    boundary_distances: np.ndarray,
    semantic_segmentation: np.ndarray,
    id_offset: float = 0,
    min_seed_distance: int = 10,
):
    """Function to compute a watershed from boundary distances."""

    seeds, n = find_local_maxima(boundary_distances, min_seed_distance)

    if n == 0:
        return np.zeros(boundary_distances.shape, dtype=np.uint64), id_offset

    seeds[seeds != 0] += id_offset

    # calculate our segmentation
    segmentation = watershed(
        boundary_distances.max() - boundary_distances, seeds, mask=semantic_segmentation
    )

    return segmentation


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 2.2</b>: <br> Use the model to generate a predicted SDT and then use the watershed function we defined above to post-process the model output into a segmentation
# </div>

# %% tags=["task"]
idx = np.random.randint(len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

# get the model prediction
# Hint: make sure set the model to evaluation
# Hint: check the dims of the image, remember they should be [batch, channels, x, y]
unet.eval()

# remember to move the image to the device
image = ... # TODO
pred = ... # TODO

# turn image, mask, and pred into plain numpy arrays
# don't forget to remove the batch dimension.
image = ... # TODO
mask = ... # TODO
pred = ... # TODO

# Choose a threshold value to use to get the boundary mask.
# Feel free to play around with the threshold.
# hint: If you're struggling to find a good threshold, you can use the <code style="color: black">threshold_otsu</code> function

threshold = ... # TODO

# Get a semantic segmentation by thresholding your distance transform
semantic_segmentation = ... # TODO

# Get the segmentation
seg = watershed_from_boundary_distance(...) # TODO

# %% tags=["solution"]
idx = np.random.randint(len(val_data))  # take a random sample
image, mask = val_data[idx]  # get the image and the nuclei masks

# get the model prediction
# Hint: make sure set the model to evaluation
# Hint: check the dims of the image, remember they should be [batch, channels, x, y]
# Hint: remember to move model outputs to the cpu and check their dimensions (as you did in task 1.4 visualization)
unet.eval()

image = image.to(device)
pred = unet(torch.unsqueeze(image, dim=0))

image = np.squeeze(image.cpu())
mask = np.squeeze(mask.cpu().numpy())
pred = np.squeeze(pred.cpu().detach().numpy())

# Choose a threshold value to use to get the boundary mask.
# Feel free to play around with the threshold.
threshold = threshold_otsu(pred)
print(f"Foreground threshold is {threshold:.3f}")

# Get inner mask
semantic_segmentation = pred > threshold

# Get the segmentation
seg = watershed_from_boundary_distance(
    pred, semantic_segmentation, min_seed_distance=20
)

# %%
# Visualize the results

plot_four(image, mask, pred, seg, label="Target", cmap=label_cmap)

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 2 </h2>

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 3: Evaluation
# Many different evaluation metrics exist, and which one you should use is dependant on the specifics of the data.
#
# [This website](https://metrics-reloaded.dkfz.de/problem-category-selection) has a good summary of different options.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 3.1</b>: Take some time to explore metrics-reloaded. This is a very helpful resource for understanding evaluation metrics.
#
# Which of the following should we use for our dataset?:
#   1) [IoU](https://metrics-reloaded.dkfz.de/metric?id=intersection_over_union)
#   2) [Accuracy](https://metrics-reloaded.dkfz.de/metric?id=accuracy)
#   3) [Sensitivity](https://metrics-reloaded.dkfz.de/metric?id=sensitivity) and [Specificity](https://metrics-reloaded.dkfz.de/metric?id=specificity@target_value)
# </div>

# %% [markdown]
# We will use Accuracy, Specificity/Precision, and Sensitivity/Recall as our evaluation metrics. IoU is also a good metric to use, but it is more commonly used for semantic segmentation tasks.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 3.2</b>: <br> Evaluate metrics for the validation dataset. Fill in the blanks
# </div>

# %% tags=["task"]
from local import evaluate

# Need to re-initialize the dataloader to return masks in addition to SDTs.
val_dataset = SDTDataset("tissuenet_data/test", return_mask=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=NUM_THREADS
)
unet.eval()

(
    precision_list,
    recall_list,
    accuracy_list,
) = (
    [],
    [],
    [],
)
for idx, (image, mask, sdt) in enumerate(tqdm(val_dataloader)):
    image = image.to(device)
    pred = unet(image)

    image = np.squeeze(image.cpu())
    gt_labels = np.squeeze(mask.cpu().numpy())
    pred = np.squeeze(pred.cpu().detach().numpy())

    # feel free to try different thresholds
    thresh = ... # TODO

    # get boundary mask
    semantic_segmentation = ... # TODO
    pred_labels = ... # TODO
    precision, recall, accuracy = evaluate(gt_labels, pred_labels)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)

print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean Accuracy is {np.mean(accuracy_list):.3f}")

# %% tags=["solution"]
from local import evaluate

# Need to re-initialize the dataloader to return masks in addition to SDTs.
val_dataset = SDTDataset("tissuenet_data/test", return_mask=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=NUM_THREADS
)
unet.eval()

(
    precision_list,
    recall_list,
    accuracy_list,
) = (
    [],
    [],
    [],
)
for idx, (image, mask, sdt) in enumerate(tqdm(val_dataloader)):
    image = image.to(device)
    pred = unet(image)

    image = np.squeeze(image.cpu())
    gt_labels = np.squeeze(mask.cpu().numpy())
    pred = np.squeeze(pred.cpu().detach().numpy())

    # feel free to try different thresholds
    thresh = threshold_otsu(pred)

    # get boundary mask
    semantic_segmentation = pred > thresh

    pred_labels = watershed_from_boundary_distance(
        pred, semantic_segmentation, id_offset=0, min_seed_distance=20
    )
    precision, recall, accuracy = evaluate(gt_labels, pred_labels)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)

print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean Accuracy is {np.mean(accuracy_list):.3f}")

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 3 </h2>
# If you reached an accuracy of about 0.4, that is good enough for this exercise. You could definitely get better results if you spend some time improving your training pipeline with any or all of: augmentations, a larger unet, more epochs.

# %% [markdown]
# <hr style="height:2px;">
#
# ## Section 4: Affinities

# %% [markdown]
# <i>What are affinities? </i><br>
# Affinities are a generalization of the a topic that we briefly touched on while computing the signed distance transform.
# Remember how we created a binary mask defining the boundaries between objects?
#
# ### 1D
# ```
# a a a b b c c c
#   a a a b b c c c
#   1 1 0 1 0 1 1
# ```
# and for visualization we can center the affinities:
# ```
# a a a b b c c c
#  1 1 0 1 0 1 1
# ```
#
#
# In this example we are shifting the labels by 1 pixel and comparing them to the original labels.
# We call this an affinity with offset 1. We can also compute affinities with different neighborhoods.
# For example: an offset of 2 might look like this:
# ```
# a a a b b c c c
#     a a a b b c c c
#     1 0 0 0 0 1
# ```
# And centering for visualization:
# ```
# a a a b b c c c
#   1 0 0 0 0 1
# ```
# Notice that we are just lengthening the boundary between the objects. Object b that is only 2 pixels wide no longer has any positive affinities :'(
#
# ### 2D
# In 2D, we can compute affinities in the same way. We can compute affinities in the x and y directions, as well as diagonally.
# Consider the offset (1,1). I'll use "-" to represent some padding for easier visualization.
# ```
# a a a b b -
# a a a b b -
# c c c b b -
# c c c b b -
# - - - - - -
# ```
# ```
# - - - - - -
# - a a a b b
# - a a a b b
# - c c c b b
# - c c c b b
# ```
# ```
# - - - - - -
# - 1 1 0 1 -
# - 0 0 0 1 -
# - 1 1 0 1 -
# - - - - - -
# ```
# Now lets look at some real affinities. In the next image we have computed 2 different offsets for our affinities. The first set with offsets (0,1) and (1,0), and the second set with offsets (0,5) and (5,0).
# Note we call a collection of offsets our affinity "neighborhood".

# %% [markdown]
# ![image](static/figure3/instance_affinity.png)

# %% [markdown]
# What do the colors mean in the affinities plots? We are displaying the affinities as RGB values. But we only have 2 offsets in our neighborhood, so those get assigned R and B. The G channel is faked by setting it to 0 only if both R and B are 0. This makes the background black, and pixels with both x and y affinity (inside our objects) show up white. The exceptions are at the boundaries where we now have either RG or BG (magenta and cyan) representing pixels that are on either the right or bottom boundaries of an object.

# %% [markdown]
# Note that the boundaries only show up the side of each object. This is because we usually don't bother centering the affinities. Half voxel shifts would add a lot of unnecessary complexity with unclear benefits to training.

# %% [markdown]
# The pros and cons of Affinities:
# - Pros:
#     - No limitations on shape or size. Blobby objects like nuclei, tiny objects only a pixel wide, and massive objects that could never fit in a field of view, all can be turned into affinities and back perfectly.
#     - Easy post-processing. We can do a simple watershed to get an initial segmentation. And then we can compute the average affinity between each pair of labels. This is very powerful for processing huge objects in small pieces.
# - Cons:
#     - Extreme class imbalance. The affinities are only 0 between objects and in background. This means networks can often learn to predict 1s everywhere and be mostly correct. Thus we always use some form of weighted loss when training affinities.
#     - Post processing can be sensitive to small errors. A naive agglomeration may merge fragments if a single high affinity edge is predicted between two separate objects, despite potentially thousands of low affinity edges contradicting this signal.

# %% [markdown]
# Similar to the pipeline used for SDTs, we first need to modify the dataset to produce affinities.

# %%
# create a new dataset for affinities
from local import compute_affinities


class AffinityDataset(Dataset):
    """A PyTorch dataset to load cell images and nuclei masks"""

    def __init__(
        self,
        root_dir,
        transform=None,
        img_transform=None,
        return_mask=False,
        weights: bool = False,
        neighborhood=None,
    ):
        self.neighborhood = (
            neighborhood
            if neighborhood is not None
            else [[0, 1], [1, 0], [0, 5], [5, 0]]
        )
        self.weights = weights
        self.root_dir = root_dir  # the directory with all the training samples
        self.num_samples = len(os.listdir(self.root_dir)) // 3  # list the samples
        self.return_mask = return_mask
        self.transform = (
            transform  # transformations to apply to both inputs and targets
        )
        self.img_transform = img_transform  # transformations to apply to raw image only
        #  transformations to apply just to inputs
        inp_transforms = transforms_v2.Compose(
            [
                transforms_v2.Normalize([0.5], [0.5]),  # 0.5 = mean and 0.5 = variance
            ]
        )
        self.from_np = transforms_v2.Lambda(lambda x: torch.from_numpy(x))

        self.loaded_imgs = [None] * self.num_samples
        self.loaded_masks = [None] * self.num_samples
        for sample_ind in tqdm(range(self.num_samples)):
            img_path = os.path.join(self.root_dir, f"img_{sample_ind}.tif")
            image = self.from_np(tifffile.imread(img_path))
            self.loaded_imgs[sample_ind] = inp_transforms(image)
            mask_path = os.path.join(self.root_dir, f"img_{sample_ind}_cyto_masks.tif")
            mask = self.from_np(tifffile.imread(mask_path))
            self.loaded_masks[sample_ind] = mask

    # get the total number of samples
    def __len__(self):
        return self.num_samples

    # fetch the training sample given its index
    def __getitem__(self, idx):
        # We'll be using the Pillow library for reading files
        # since many torchvision transforms operate on PIL images
        image = self.loaded_imgs[idx]
        mask = self.loaded_masks[idx]
        if self.transform is not None:
            # Note: using seeds to ensure the same random transform is applied to
            # the image and mask
            seed = torch.seed()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        aff_mask = self.create_aff_target(mask)
        if self.img_transform is not None:
            image = self.img_transform(image)

        if self.weights:
            weight = torch.zeros_like(aff_mask)
            for channel in range(weight.shape[0]):
                weight[channel][aff_mask[channel] == 0] = np.clip(
                    weight[channel].numel()
                    / 2
                    / (weight[channel].numel() - weight[channel].sum()),
                    0.1,
                    10.0,
                )
                weight[channel][aff_mask[channel] == 1] = np.clip(
                    weight[channel].numel() / 2 / weight[channel].sum(), 0.1, 10.0
                )

            if self.return_mask is True:
                return image, mask, aff_mask, weight
            else:
                return image, aff_mask, weight
        else:
            if self.return_mask is True:
                return image, mask, aff_mask
            else:
                return image, aff_mask

    def create_aff_target(self, mask):
        aff_target_array = compute_affinities(np.asarray(mask), self.neighborhood)
        aff_target = torch.from_numpy(aff_target_array)
        return aff_target.float()


# %% [markdown]
# Next we initialize the datasets and data loaders.

# %%
# Initialize the datasets

# TODO: feel free to play around with the neighborhood parameter
# The visualization code will break if you change the number of affinities, but feel free to change the magnitudes.
# Training will break if you change the number of affinities. It is a simple fix, you will just need to change the number
# of output channels the unet produces.

neighborhood = [[0, 1], [1, 0], [0, 5], [5, 0]]
train_data = AffinityDataset(
    "tissuenet_data/train",
    transforms_v2.RandomCrop(128),
    weights=True,
    neighborhood=neighborhood,
)
train_loader = DataLoader(
    train_data, batch_size=5, shuffle=True, num_workers=NUM_THREADS
)
idx = np.random.randint(len(train_data))  # take a random sample
img, affinity, weight = train_data[idx]  # get the image and the nuclei masks
plot_two(img, affinity, label="Affinities")

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Task 4.1</b>: Train a model with affinities as targets.
#
# Repurpose the training loop which you used for the SDTs. <br>
# Think carefully about your final activation and number of out channels. <br>
# (The best for SDT is not necessarily the best for affinities.)
# </div>

# %% tags=["task"]

unet = ... # TODO
learning_rate = ... # TODO
# Note you will need to use <code style="color: black">reduce=False</code> for whatever loss function you choose. The easiest choices will be <code style="color: black">BCELoss</code> or <code style="color: black">MSELoss</code>.
# Normally for e.g. MSE loss you compute the squared error of each pixel, then reduce with the mean, and backpropogate.
# However we want to weight each pixel separately, so we compute the loss per pixel, then multiply by that pixels weight, then reduce with the mean.
# This provides a larger gradient for pixels that have a larger weight since they will contribute more the the final loss.
loss = ... # TODO
optimizer = ... # TODO

# %% tags=["solution"]

unet = UNet(
    depth=4,
    in_channels=2,
    out_channels=len(neighborhood),
    final_activation=torch.nn.Sigmoid(),
    num_fmaps=16,
    fmap_inc_factor=3,
    downsample_factor=2,
    padding="same",
)

learning_rate = 1e-4

# choose a loss function
loss = torch.nn.BCELoss(reduce=False)

optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)

# %% tags=["task"]
for epoch in range(NUM_EPOCHS):
    train(
        ..., # TODO
        ..., # TODO
        ..., # TODO
        ..., # TODO
        ..., # TODO
        log_interval=2,
        device=device,
    )

# %% tags=["solution"]
for epoch in range(NUM_EPOCHS):
    train(
        unet,
        train_loader,
        optimizer,
        loss,
        epoch,
        log_interval=2,
        device=device,
    )

# %% [markdown]
# Let's next look at a prediction on a random image.
# We will be using mutex watershed (see this paper by [Wolf et al.](https://arxiv.org/abs/1904.12654)) for post processing. I won't dive too much into the details, but it is similar to watershed except that it allows edges to have negative weights for handling split errors, and removes the need for finding seed points and having minimum seed distance as a parameter.
#
#
# However this does mean we now need a bias term since if we give it all positive edges (our affinities are in range (0, 1)) everything will be merged into a single object. Thus our bias should be in range (-1, 0), such that we have some positive and some negative affinities.
#
#
# It can also be useful to bias long range affinities more negatively than the short range affinities. The intuition here being that boundaries are often blurry in biology. This means it may not be easy to tell if the neighboring pixel has crossed a boundary, but it is reasonably easy to tell if there is a boundary accross a 5 pixel gap. Similarly, identifying if two pixels belong to the same object is easier, the closer they are to each other. Providing a more negative bias to long range affinities means we bias towards splitting on low long range affinities, and merging on high short range affinities.

# %%
val_data = AffinityDataset(
    "tissuenet_data/test", transforms_v2.RandomCrop(128), return_mask=True
)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)

unet.eval()
idx = np.random.randint(len(val_data))  # take a random sample
image, gt, affs = val_data[idx]  # get the image and the nuclei masks
image = image.to(device)
pred = torch.squeeze(unet(torch.unsqueeze(image, dim=0)))
image = image.cpu()
affs = affs.cpu().numpy()
pred = pred.cpu().detach().numpy()
gt_labels = gt.cpu().numpy()

bias_short = -0.9
bias_long = -0.95

pred_labels = mws.agglom(
    np.array(
        [
            pred[0] + bias_short,
            pred[1] + bias_short,
            pred[2] + bias_long,
            pred[3] + bias_long,
        ]
    ).astype(np.float64),
    neighborhood,
)

# Mutex watershed often leads to many tiny fragments due to the fuzziness of our models predictions.
# We can add a simple small object filter to get significantly higher accuracy.
precision, recall, accuracy = evaluate(gt_labels, pred_labels)
print(
    f"Before filter: Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}"
)
plot_four(image, affs, pred, pred_labels, label="Affinity", cmap=label_cmap)
pred_labels = remove_small_objects(
    pred_labels.astype(np.int64), min_size=64, connectivity=1
)
precision, recall, accuracy = evaluate(gt_labels, pred_labels)
print(
    f"After filter: Precision: {precision:.3f}, Recall: {recall:.3f}, Accuracy: {accuracy:.3f}"
)
plot_four(image, affs, pred, pred_labels, label="Affinity", cmap=label_cmap)

# %% [markdown]
# Let's also evaluate the model performance.

# %%

val_dataset = AffinityDataset("tissuenet_data/test", return_mask=True)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, num_workers=NUM_THREADS
)
unet.eval()

(
    precision_list,
    recall_list,
    accuracy_list,
) = (
    [],
    [],
    [],
)
for idx, (image, mask, _) in enumerate(tqdm(val_dataloader)):
    image = image.to(device)

    pred = unet(image)

    image = np.squeeze(image.cpu())

    gt_labels = np.squeeze(mask.cpu().numpy())

    pred = np.squeeze(pred.cpu().detach().numpy())

    pred_labels = mws.agglom(
        np.array(
            [
                pred[0] + bias_short,
                pred[1] + bias_short,
                pred[2] + bias_long,
                pred[3] + bias_long,
            ]
        ).astype(np.float64),
        neighborhood,
    )
    pred_labels = remove_small_objects(
        pred_labels.astype(np.int64), min_size=64, connectivity=1
    )

    precision, recall, accuracy = evaluate(gt_labels, pred_labels)
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)

print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean Accuracy is {np.mean(accuracy_list):.3f}")

# %% [markdown]
# <hr style="height:2px;">
#
# ## Bonus: Further reading on Affinities
# [Here](https://localshapedescriptors.github.io/) is a blog post describing the Local Shape Descriptor method of instance segmentation.
#

# %% [markdown]
# <div class="alert alert-block alert-success">
# <h2> Checkpoint 4 </h2>
# You have now completed the exercise! If you achieved an accuracy of around 0.5, that is pretty decent. If you have time to spare you can try ot push the accuracy higher. I would recommend exploring augmentations (rotations work very nicely for this task), a larger unet, and more epochs, and potentially adjusting the learning rate.

# %% [markdown]
# <hr style="height:2px;">
#
# ## Bonus: Pre-Trained Models
# Cellpose has an excellent pre-trained model for instance segmentation of cells and nuclei.
# <br> take a look at the full built-in models and try to apply one to the dataset used in this exercise.
# <br> -[cellpose github](https://github.com/MouseLand/cellpose)
# <br> -[cellpose documentation](https://cellpose.readthedocs.io/en/latest/)
#
#

# %%
# cellpose was installed by setup.sh (v3 pinned in requirements.txt).

# %%
from cellpose import models

model = models.CellposeModel(pretrained_model="cyto3", device=device)
channels = [[0, 0]]

# %% [markdown]
# Let's first look at one cellpose prediction next to the ground truth.

# %%
idx = np.random.randint(len(val_loader.dataset))
image, mask, _ = val_loader.dataset[idx]
image = image.cpu().numpy()
gt_labels = np.squeeze(mask.cpu().numpy())
masks, flows, _ = model.eval([image], diameter=None, channels=channels)
pred_labels = masks[0]
flow_rgb = flows[0][0]  # cellpose flow visualization

# render GT through label_cmap so plot_four uses it (intermediate panel defaults to coolwarm)
gt_rgb = label_cmap(gt_labels)[..., :3].astype(np.uint8)

plot_four(image, gt_rgb, flow_rgb, pred_labels, label="GT", cmap=label_cmap)

# %% [markdown]
# Now over the full validation set.

# %%
precision_list, recall_list, accuracy_list = [], [], []
for idx, (image, mask, _) in enumerate(tqdm(val_loader)):
    gt_labels = np.squeeze(mask.cpu().numpy())
    image = np.squeeze(image.cpu().numpy())
    pred_labels, _, _ = model.eval([image], diameter=None, channels=channels)

    precision, recall, accuracy = evaluate(gt_labels, pred_labels[0])
    precision_list.append(precision)
    recall_list.append(recall)
    accuracy_list.append(accuracy)

print(f"Mean Precision is {np.mean(precision_list):.3f}")
print(f"Mean Recall is {np.mean(recall_list):.3f}")
print(f"Mean Accuracy is {np.mean(accuracy_list):.3f}")

# %% [markdown]
#
