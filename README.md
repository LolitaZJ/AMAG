# AMAG
Local Magnitude Estimation via An Attention-based Machine Learning Model
Author: Ji Zhang  
Date: 2024.06.14 
Version 1.0.0  

# AMAGNET   
## Local Magnitude Estimation via An Attention-based Machine Learning Model

### This repository contains the codes to train and test the network proposed in:             

`Ji, Zhang, Kato, A., Zhu H., & Wang, W. (2024). Local Magnitude Estimation via An Attention-based Machine Learning Model (submitted).`
      
------------------------------------------- 
### Installation:

   `pip install -r requirements.txt`

or

   `pip install keras-self-attention`
   
------------------------------------------- 
### Short Description:

 Rapid and reliable earthquake magnitude estimation is crucial for earthquake early warning (EEW), especially during the initial stages of event detection. Traditional methods rely on complete waveform records, including earthquake epicenter distance and waveform amplitude, which can delay magnitude assessment. Machine learning techniques offer a promising avenue for capturing nonlinear relationships within seismic data, enhancing both information extraction and timeliness in magnitude estimation. We introduce an Attention-based Machine Learning model for Magnitude Estimation (AMAG) tailored for EEW applications.

------------------------------------------- 
### Dataset:

Training Data from [STEAD](https://github.com/smousavi05/STEAD) or `https://github.com/smousavi05/STEAD`

Testing Ddata from [PNW](https://github.com/niyiyu/PNW-ML) or `https://github.com/niyiyu/PNW-ML`

or You can use [INSTANCE](https://github.com/niyiyu/PNW-ML) data to train model

------------------------------------------- 
### Run
First, you need to set --data_path your/hdf5/ path --csv_path your/csv/path. 

`Fast Train and Test `
`to see all configure is right, the model may not well because of less data`
>     nohup python Atten_EEW_MASTER_ALL.py --data_path your/hdf5/path --csv_path your/csv/path --batch_size=1024 --loss=mse --save_name=AMAG_d4k5 --depth=4 --kernel_size=5 --ratio=0 --mode=train --GPU=3 --epoch=  > AMAG_d4k5.txt 2>&1 &

--loss mse: choose mse as Loss function;     
--depth 4: set the depth of network 4;   
--kernel_size 5: set the kernel size 5;    
--GPU 3: if you don't have GPU, set -1;

`Train all datasets and Test`
>     nohup python Atten_EEW_MASTER_ALL.py --fast False --data_path your/hdf5/path --csv_path your/csv/path --batch_size=1024 --loss=mse --save_name=AMAG_d4k5 --depth=4 --kernel_size=5 --ratio=0 --mode=train --GPU=3 > AMAG_d4k5.txt 2>&1 &

If you want to use INSTANCE to train a model
set --stead Fasle and --data_path your/instance/hdf5/ path --csv_path your/instance/csv/path. 

`INSTANCE`
>     nohup python Atten_EEW_MASTER_ALL.py --stead False --data_path your/instance/hdf5/ path --csv_path your/instance/csv/path --batch_size=1024 --loss=mse --save_name=AMAG_d4k5 --depth=4 --kernel_size=5 --ratio=0 --mode=train --GPU=3 > AMAG_d4k5.txt 2>&1 &

------------------------------------------- 
### Model:

You can use the model **AMAG_d4k5.h5** trained on the STEAD dataset to predict magnitude.
You may need to re-train or refine the model with your own dataset. 

------------------------------------------- 
