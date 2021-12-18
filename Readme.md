[toc]

## Pipeline

![Pipeline](./docs/images/Pipeline.png)

## File Structure

+ components

  + models 
  + dataset_loader
  + train_strategy
  
+ useful_functions

  + losses
  + metrics

+ utils

+ config_scripts

+ results

  + logs
  + checkpoints 

+ tools

  

## Add self-defined components

Below is an instruction for adding a self-defined model for use.

### 1 Add indexes  for  your self-defined item

Add self-defined items in `utils/supported_items.py`. This file has 4 dictionaries:

+ supported_model_dict 
+ supported_loss_dict 
+ supported_dataloader_dict 
+ supported_training_strategy_dict 

Fill them with your method.

### 2 Add your method in corresponding folder 

Add your self-defined methods or functions in the corresponding folder for clear organization.

+ Put you model in `components/models` folder 
+ Put you data loader in `components/dataset_loader` folder 
+ Put you train strategy in `components/train_strategy` folder 
  + Optimization, Loss Function, Metrics are included in a training strategy
  + You may would like to use some self-defined loss functions. Write them directly in `train_strategy` or put then  in  `useful_functions/losses` folder 
  + You may would like to use some self-defined metrics to evaluate your model results. Write them directly in `train_strategy` or put then  in  `useful_functions/losses` folder 

This framework also provide some template for easy use:

+ Template for data loader in `components/train_strategy/dataset_loader_template.py`
+ Template for training strategy in `components/train_strategy/train_strategy_template.py`

### 3 Write your configuration files 

All your configurations can be summarize in one `yaml` file, so that you can change some configurations easily. The configuration file is located at `config_scripts` folder. There is also a template provided: `config_scripts/template.yml`.

## Model Explanation

> **Symbol Explanation**
>
> + $C = Input \ Channel$
> + $O = Output \ Channel$
> + $B = Batch \ Size$
> + $H = Height$
> + $W = Width$

| Model Name | Special Symbols                             | Explanations | Input Shape       | Output Shape      |
| ---------- | ------------------------------------------- | ------------ | ----------------- | ----------------- |
| UNet       |                                             | 2D UNet      | $[B, C, H, W]$    | $[B, O, H, W]$    |
| UNet_Gan   |                                             | 2D UNet Gan  | $[B, C, H, W]$    |                   |
| UNet3D     | $S = Image \ Stack \ Num, where \  S \ge 8$ | 3D UNet      | $[B, C, H, W, S]$ | $[B, O, H, W, S]$ |
|            |                                             |              |                   |                   |
|            |                                             |              |                   |                   |

