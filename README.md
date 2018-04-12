# CycleGAN_DA
Using CycleGAN to do domain adaptation between simulation and real world.

## Usage: 
        1) build data: python build_data.py --X_input_dir "sim_images" --X_output_file "sim_images_a1_low_var.tfrecords"
        2)run the network: python train_CycleGAN.py --X "tfrecordsname"
        3)open tensorboard: tensorboard --logdir checkpoints/${datetime}
         4)export trained network: python export_graph_CycleGAN.py --checkpoint_dir checkpoints/20180410-2134 --XtoY_model sim2real.pb  --YtoX_model real2sim.pb
        5)inference: python inference_CycleGAN.py --model pretrained/sim2real.pb --input 000000.jpeg --output output_sample.jpg



## Structure:
```
CycleGAN_DA
│   README.md
│   ops_CycleGAN.py  
│   discriminator_CycleGAN.py
│   generator_CycleGAN.py 
│   model_CycleGAN.py
│   inference_CycleGAN.py
│   reader_CycleGAN.py
│   export_graph_CycleGAN.py
│   utils.py 
│   build_data.py
│   train_CycleGAN.py
│   utils_CycleGAN.py
│   sim_images_a1_low_var.tfrecords  
│   ...
│
└───Data
│   │   features_060.csv
│   │   features_097.csv
│   │
│   └───tfdata
│       │   grasping_dataset_060.tfrecord-00000-of-00022
│       │   grasping_dataset_060.tfrecord-00001-of-00022
│       │   ...
│       │
│   └───tfdata1
│       │   grasping_dataset_097.tfrecord-00000-of-00034
│       │   grasping_dataset_097.tfrecord-00001-of-00034
│       │   ...
│   
└───random_urdfs
│   │
│   └───000
│       │   000_coll.mtl
│       │   000.urdf
│       │   ...
│       │
│   └───001
│       │   001_coll.mtl
│       │   00.urdf
│       │   ...
│       │
│   └───...
│       │
│       │
│       
└───sim_images
│   │
│   │   sim_image000000.jpeg
│   │   sim_image000001.jpeg
│   │   ...
│   │
└───checkpoints
│   │
│   └───20180331-0000
│       │   checkpoint
│       │   ...
│       │
│   └───20180331-0005
│       │   checkpoint
│       │   ...
```
    


