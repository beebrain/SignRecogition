# SignRecogition
This is a Sign Recognition(SR) State in paper ... and ...
This code will train and test a signing image with CNN. It save the model automatily.

How to run
1. Run RandomHandclass.py file to unpack the training dataset and testing dataset.
  1.0 Please download the sign image package files (25signHand_size256_dataset_Test,25signHand_size256_dataset_Train) on Google drive. Please followed this link https://drive.google.com/drive/folders/1yIvGRdFKHbCqsfPqKzaC-feZz1TguVEh?usp=sharing.
  1.1 This script will create the folder of training image, Testing image, and validation image. The training images are contained in ClassImage_train folder, the testing image are contrined in ClassImage_test, and the folder ClassImage_val contain validation image.
  
2. Run CreateModel.py to train the model that will give information about training acc and testing acc. While training will save the parameter that give the best accuracy. The model and weight parameter will save in Model_CNN folder.

