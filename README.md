# Fear ReFactor
Fear ReFactor is a web app that masks phobia objects on YouTube videos. Currently, the specific objects, 'clown', 'dog', 'bird', and 'teddy bear' are available.
Notice that Fear ReFactor requires GPU to run properly. If you prefer to run this streamlit app in your local computer, use [this code](https://github.com/HannahhoHe/Fear-ReFactor-Mask-R-CNN-Transfer-Learning/blob/master/FearReFactor_streamlit_pub.py) and don't forget to type in your email login credentials (line 526 and line 535). Otherwise, this streamlit web app is available [here](https://52.34.156.240:8501), deployed on Amazon EC2 (p2 instance). Please email me Dr.HeHannah@gmail.com if the port is not open.   

This repo focus on building a model to detect and mask ALL 'clown' objects throughout a YouTube video. To achieve this, I performed a transfer learning with [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN). Below is a post-masked clown video from [YouTube](https://www.youtube.com/watch?v=GGOMD2DlJUY&t=107s).  

<p align="center">
  <img align="middle" width="450" src="gif_small.gif">
</p>

# Workflow
Fear ReFactor takes every frame from the video, running through the Mask R-CNN models built in this repo, and then re-constructs all the processed frames to a 3D video, stroed in Amazon S3 bucket. All the processed video frames should have every clown object completely masked. [This code](https://github.com/HannahhoHe/Fear-ReFactor-Mask-R-CNN-Transfer-Learning/blob/master/vid-im.ipynb) includes the process of parsing image frames and audio from a YouTube video, and reconstructing and playing the final video.    

<p align="center">
  <img align="middle" width="450" src="flow.PNG">
</p>
