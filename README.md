## Adaptive adversarial 
### Introduction
This project is the code for paper Adaptive Adversarial samples based Active learning for medical image classification based on python and pytorch framework.
  

### Requirements  
The main package and version of the python environment are as follows
```
# Name                    Version         
python                    3.8.3                    
pytorch                   1.10.1         
torchvision               0.11.2         
cudatoolkit               10.2.89       
cudnn                     7.6.5           
matplotlib                3.6.2              
numpy                     1.19.2        
opencv                    4.4.0        
pandas                    1.4.4               
scikit-learn              1.2.0               
tqdm                      4.64.1             
```  

The above environment is successful when running the code of the project. Pytorch has very good compatibility. Thus, I suggest that try to use the existing pytorch environment firstly.

---  
## Usage 
### 1) Download Project 

Running```git clone https://github.com/activelearning2022/adversarial_active_learning.git```  
The project structure and intention are as follows : 
```
Adversarial active learning			# Source code		
    ├── seed.py			 	                                          # Set up random seed
    ├── query_strategies		                                    # All query_strategies
    │   ├── adaptive_adversarial_sample.py                          # Our method
    │   ├── bayesian_active_learning_disagreement_dropout.py	  # Deep bayesian query method
    │   ├── entropy_sampling.py		                              # Entropy based query method
    │   ├── entropy_sampling_dropout.py		                      # Entropy based MC dropout query method
    │   ├── kcenter_greedy.py		                              # Coreset
    │   ├── random_sampling.py		                              # Random selection
    │   ├── strategy.py                                         # Functions needed for query strategies
    ├── data.py	                                                # Prepare the dataset & initialization and update for training dataset
    ├── data_func.py	                                                # Main functions used for data processing
    ├── handlers.py                                             # Get dataloader for the dataset
    ├── main.py			                                            # An example for code utilization, including the whole process of active learning
    ├── nets.py		                                          # Training models and methods needed for query method
    ├── supervised_baseline.py	                                # An example for supervised learning traning process
    └── utils.py			                        # Important setups including network, dataset, hyperparameters...
```
### 2) Datasets preparation 
1. Download the datasets from the official address:
   
   MSSEG: https://portal.fli-iam.irisa.fr/msseg-challenge

   
2. Modify the data folder path for specific dataset in `data.py`

### 3) Run Active learning process 
Please confirm the configuration information in the [`utils.py`]
```
  python main.py \
      --n_round 25 \
      --n_query 500 \
      --n_init_labeled 1000 \
      --dataset_name MSSEG \
      --traning_method supervised_val_loss \
      --strategy_name RandomSampling \
      --seed 42
```
The training results will be saved to the corresponding directory(save name) in `performance.csv`.  
You can also run `supervised_baseline.py` by
```
python supervised_baseline.py
```

## Visualization
1 Active learning query sample distribution visualization  
After you got the `performance.csv`, you can run `visualization.py` to visualize the whole process

