# CAP-5516-Course_Project
Guided Synthesis of MRI images using a C-GAN


Provided arguments:

Train_Generator: defaults to false. When set to true, will retrain the generator and discriminator, then save the params for the generator to the trained_generator_params file

Generate_Images: Generates images for both the train and evaluation dataset. Defaults to True. The training dataset will have 100 synthetic images of each class; the evaluation dataset will have 50.

Train_Classifiers: Defaults to False. When true, will train 5 different classifiers with a ResNet-18 backbone: 1. trained purely on the real images. 2. Trained on the real images and the synthetic healthy images. 3. Trained on the real images and the synthetic MCI images. 4. Trained on the real images and the synthetic AD images. 5. Trained on the real images, and the synthetic unhealthy images (MCI and AD). 6. Trained on the real images, and all of the synthetic images. The parameters at the end of training are saved.

Evaluate_Classifiers: Defaults to True. Evaluates the above classifiers on data not seen before during training, consisting of real and synthetic images of all classes. Outputs accuracy, recall on MCI images, and recall on AD images for each model.