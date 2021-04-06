# image-segmentation-U-Net
[Data source: kaggle 2018 data science bowl](https://www.kaggle.com/c/data-science-bowl-2018)
# Netwrok built based on mentionde U-net structure
![](https://github.com/tural327/image-segmentation-U-Net/blob/master/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313630302f312a54586645507154624642504362585968326273746c412e706e67.png)



# Training preparation
Before starting training we shuld make ready training dataset to fit in our model. In traing folder we had 670 folder and each folder has own image file named folder name and mask pictures in that case our for first step we should make our [training file](https://github.com/tural327/image-segmentation-U-Net/blob/master/training_file.py) ,  while making it i used for loop and added my data set X = np.zeros((len(folder), img_h, img_w, img_c), dtype=np.uint8) file beacuse i need input data for our modul shape of (670,128,128,3) and for Y (validation) first i merged mask file and as X(input) file was added [Y = np.zeros((len(folder), img_h, img_w, 1), dtype=np.bool)](https://github.com/tural327/image-segmentation-U-Net/blob/master/training_file.py)

# Training our model 

In this section model was built. First layers was same like simple CNN but from buttom of our sturcture model strating transpose and making connection between pervious CNN layers

[Please check training file:](https://github.com/tural327/image-segmentation-U-Net/blob/master/model_u_net.py)



for validation our network i used 0.2 % of my dataset after fitting result was :

![](https://github.com/tural327/image-segmentation-U-Net/blob/master/loss_graf.png)

# Testing Section 

[I buid test file]( also for my model

[test file here (clickfor checking)](https://github.com/tural327/image-segmentation-U-Net/blob/master/testing_model.py)


