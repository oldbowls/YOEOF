# YOEOF
A General Method for Converting Offline Skeleton-based Action Recognition to Online Ones

# Note
    pytorch>=1.6

# Data Preparation
Under the "code" forder: 

 - NTU-60
    - Download the NTU-60 data from the https://github.com/shahroudy/NTURGB-D to `../data/raw/ntu60`
    - `cd prepare/ntu60/`
    - Process the raw data sequentially with `python get_raw_skes_data.py`, `python get_raw_denoised_data.py` and `python seq_transformation.py`
 - NTU-120
    - Download the NTU-120 data from the https://github.com/shahroudy/NTURGB-D to `../data/raw/ntu120`
    - `cd prepare/ntu120/`
   - Process the raw data sequentially with `python get_raw_skes_data.py`, `python get_raw_denoised_data.py` and `python seq_transformation.py`

# Training & Testing

 - Using NTU-60-CS as an example: 
 - Run backbone_main.py to train the student model, and make sure to modify the configuration file accordingly in backbone_main.py.
    - `python backbone_main.py`
 - Note that before training the student model, it is essential to train the corresponding teacher model. Refer to https://github.com/MartinXM/GAP for the process of training the teacher model.
