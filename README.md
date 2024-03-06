# General description of the project

The project investigates the impact of different data augmentation techniques on improving the accuracy of ResNet convolutional neural networks for palmprint recognition. Currently, existing ResNet convolutional neural networks exhibit suboptimal recognition accuracy, achieving only around 80% accuracy due to limited sample sizes. To address this, various common data augmentation techniques were employed to augment three original images, including adjustments to brightness, contrast, random noise addition, and image flipping, among others. Additionally, the study examines the effects of both single-method and combined-method data augmentation on ResNet convolutional neural networks.



In this experiment, a publicly available dataset was utilized. Multiple data augmentation techniques were employed in ten sets of trials, with batch sizes of 32 and 64. The best experimental result achieved was an increase in the original accuracy to 86%. This optimal outcome occurred when employing brightness adjustment, contrast adjustment, and simultaneous brightness and contrast adjustments. This suggests that for palmprint recognition tasks, employing more complex data augmentation techniques does not necessarily yield superior experimental results. Indeed, it is demonstrated that simple image enhancement methods can significantly enhance the recognition accuracy of ResNet convolutional neural networks. This experiment provides insights into data augmentation strategies for palmprint recognition tasks with limited datasets in the future.



Email：**2015yunfei@gmail.com**



# 原项目地址

https://github.com/ruofei7/Palmprint_Recognition





# 我的工作



My improvement on the original project：



In this experiment, various image augmentation techniques were employed to augment the limited training set, thereby increasing the training samples and enhancing the generalization ability of the ResNet model. Ultimately, the recognition accuracy of the ResNet model was increased from 80% to 86%.

 

I combined common data augmentation techniques, including brightness adjustment, contrast adjustment, random noise addition, random cropping, image rotation, and image flipping, into ten groups for experimentation. Through these ten sets of trials, I aimed to identify the optimal data augmentation approach.
