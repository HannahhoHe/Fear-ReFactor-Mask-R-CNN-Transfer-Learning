# Fear ReFactor
Fear ReFactor is a web app that masks phobia objects for YouTube videos. Currently, the specific objects, 'clown', 'dog', 'bird', and 'teddy bear' are available.
Notice that Fear ReFactor requires GPU to run properly. If you prefer to run the streamlit app in your local computer, use [this code](https://github.com/HannahhoHe/Fear-ReFactor-Mask-R-CNN-Transfer-Learning/blob/master/FearReFactor_streamlit_pub.py) and don't forget to type in your email login credentials (line 526 and line 535). Otherwise, the streamlit web app is available [here](https://52.34.156.240:8501), deployed on Amazon EC2 (p2 instance). Please email me Dr.HeHannah@gmail.com if the port is not open.   

This repo focuses on building a model to detect and mask ALL 'clown' objects in a video. To achieve this, I performed a transfer learning with [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).  


![alt text](gif_small.gif width="48")

[original](https://www.youtube.com/watch?v=GGOMD2DlJUY&t=107s)

