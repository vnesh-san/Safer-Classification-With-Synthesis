# Safer-Classification-With-Synthesis
This project is a part of the Full Stack Deep Learning Spring 2021 course.

# Goal: 
Develop a classification model that learns about the data distribution in addition to traditional classification task. This enables the algorithm to distinguish data points that come out-of-distribution than the data used during training. For example, a traditional classification algorithm for dog-vs-cat problem would still classify an image of a horse/rat as dog/cat with relatively high confidence. Whereas the current attempted model would be able to distinguish that the said, horse/rat image that it received during testing is out-of-distribution. 

# How it is achieved??
As the authors of the paper https://arxiv.org/abs/1711.08534 have stated, a generative model can only generate data points from a distribution it has learnt on. The authors develop a model where they train a seperate generator for each class in the training data. During testing, they query each generator to generate data points and generator_idx of the generator that produced the data point that is closest to the test data point is the corresponding predicted label of that test data point. 
This method seems to work best in identifying the test samples that are out-of-distribution because none of the generators would be able to generate data points that are closer to test sample. Such a scenario will tell us that the test sample is out-of-distribution. 

However, the bottleneck here is that the model parameters quickly scale up as the number of classes in our training dataset. To tackle this, the solution proposed and implemented here is that, we train a larger generator more on the whole training data distribution instead of having one generator for every class. Ideally, from emperical results, it is suggested that the generator be pre-trained before moving further. In addition, there is a linear layer for each class preceding the generator. Each linear layer learns to map a random noise to a latent vector which when fed to the generator as input would result in a data point from the corresponding class. The following figure depicts this better. 

<img width="692" alt="Screen Shot 2021-05-15 at 9 47 21 PM" src="https://user-images.githubusercontent.com/34956791/118385807-25aac500-b5c7-11eb-8856-0e94653796e2.png">

The rest of the procedure is maintained the same as the authors proposed (which involves either a distance metric or siamese network to identify if the generated image and the image to be classified belong to the same class or not).

The pretrained generator and siamese network weights are attached for MNIST dataset. The method could be expanded to any domain and any dataset as pleased.

# Command
To the run the experiment in the default setting, run the command,
`python run_experiment.py `

To play around and know more about the list of available parser arguments, use the command, 
`python run_experiment.py --help`
