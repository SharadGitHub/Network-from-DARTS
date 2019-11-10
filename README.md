# Network-from-DARTS
This is the implementation of the network based on best normal and reduction cells found in DARTS paper.  

# Normal cell 
Normal cell keeps the feature maps size same and channels as well. It uses depthwise seperable convolutional operations and dilation separable convolution. All operations have stride = 1, kernel size = 3 and padding to preserve the spatial size. 

![Screenshot from 2019-11-10 17-15-50](https://user-images.githubusercontent.com/14364405/68547138-2f55da00-03de-11ea-9e19-a00ffb3b751e.png)


# Reduction cell
Reduction cell mains to reduce the spatial size by half and double the number of channels of inputs. Max pooling is used to reduce the spatial size and conv layer with 1x1 kernel is used to double the channels. 


![Screenshot from 2019-11-10 17-15-50 (1)](https://user-images.githubusercontent.com/14364405/68547149-6d52fe00-03de-11ea-8ea4-e4bc46983240.png)


# Network
Normal cell and reduction cells are stacked to make a network which is trained on CIFAR10 dataset. 

![Screenshot from 2019-11-10 17-25-05](https://user-images.githubusercontent.com/14364405/68547214-30d3d200-03df-11ea-89b6-fa5190afffd2.png)

In the figure above, cell with stride 1 is normal cell and the one with stride 2 is reduction cell. In this implementation N = 2, so there are 8 layers in the network, 2 reduction cells and 6 normal cells.

As given in the paper, 1x1 convolutional operations are inserted to make the inputs compatible for further operations in the cells. 
