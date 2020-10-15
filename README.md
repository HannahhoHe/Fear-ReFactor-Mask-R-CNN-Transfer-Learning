# What is Fear ReFactor?
Fear ReFactor is a web app that masks phobia objects on YouTube videos. Currently, the specific objects, 'clown', 'dog', 'bird', and 'teddy bear' are available.
Notice that Fear ReFactor requires GPU to run properly. If you prefer to run this streamlit app on your local computer, use [this code](https://github.com/HannahhoHe/Fear-ReFactor-Mask-R-CNN-Transfer-Learning/blob/master/FearReFactor_streamlit_pub.py) and don't forget to type in your email login credentials (line 526 and line 535). Otherwise, this streamlit web app is available [here](https://52.34.156.240:8501), deployed on Amazon EC2 (p2 instance). Please email me at Dr.HeHannah@gmail.com if the port is not open.   

This repo focuses on building a model to detect and mask ALL 'clown' objects throughout a YouTube video. To achieve this, I performed transfer learning with [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN). Below is a post-masked clown video from [YouTube](https://www.youtube.com/watch?v=GGOMD2DlJUY&t=107s).  

<p align="center">
  <img align="middle" width="450" src="gif_small.gif">
</p>

# Fear ReFactor Workflow
Fear ReFactor takes every frame from the video, running through the Mask R-CNN models built in this repo, and re-constructs all the processed frames to a video, stored in Amazon S3 bucket. Every processed video should have the clown object completely masked. [This code](https://github.com/HannahhoHe/Fear-ReFactor-Mask-R-CNN-Transfer-Learning/blob/master/vid-im.ipynb) includes the process of parsing video and audio frames, and constructing and playing a video.    

<p align="center">
  <img align="middle" width="450" src="flow.PNG">
</p>

# Building up Mask R-CNN models 
To prepare training images, I scraped static images and YouTube images. [This code](https://github.com/HannahhoHe/Fear-ReFactor-Mask-R-CNN-Transfer-Learning/blob/master/google-im.ipynb) shows how to use Google Selenium and headless Chrome in AWS EC2.    
