import streamlit as st
from youtubesearchpython import SearchVideos
import pandas as pd
from lxml import html, etree
import pafy
import json
import os
import sys 
import cv2
from os.path import isfile, join
import IPython.display
from PIL import Image
from ffpyplayer.player import MediaPlayer
from pytube import YouTube
from moviepy.editor import *
from moviepy.video.VideoClip import VideoClip
from os.path import isfile, join
from youtubesearchpython import SearchVideos
import warnings
warnings.simplefilter("ignore")
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from numpy import expand_dims
from numpy import mean
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append('/home/ubuntu/Mask_RCNN/samples/coco') #adjust 
import coco
import itertools
import colorsys
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from PIL import Image
from mrcnn import visualize
from ffpyplayer.player import MediaPlayer
from pytube import YouTube
from moviepy.editor import *
import smtplib
from email.mime.text import MIMEText
import moviepy.editor as mpe
import boto3



st.header('Fear ReFactor')
st.markdown("### 🎲 Enter YouTube URL")
url = st.text_input("", key = 'weburl')
video = pafy.new(url)
bestResolutionVideo = video.getbest()
st.write(f'Title:  {bestResolutionVideo.filename}')

st.markdown("### 🎲 What are afraid of?")
phobia = st.selectbox("Select your phobia: ", ['clown', 'dog', 'teddy bear', 'bird'])

st.markdown("### 🎲 Select Video Fragment" + "  ")
sec1 = st.number_input("START /Sec", value=50, key = 0)
sec2 = st.number_input('END /Sec', value=65, key = 1)
minutes = round((((sec2 - sec1)*30)/0.5)/60)
st.write(f'Estimate: {minutes} minutes')



fps = 30  #fps = 30
MaxCount = (sec2 - sec1)*30  #30
n_images = MaxCount-1 


# This part is running R-CNN
##################################################################################################################################

def download():
    video = pafy.new(url)
    bestResolutionVideo = video.getbest()
    bestResolutionVideo.download()



# audio edits
def audio(sec, MaxCount):
    a = sec1
    b = sec2
    audio_cut = VideoFileClip(bestResolutionVideo.filename).subclip(a,b)
    audioclip = audio_cut.audio
    audioclip.write_audiofile('audio.mp3')



# Run MRCNN 

class_names2 = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
           'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
           'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
           'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
           'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
          'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
          'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
           'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush'] 

class_names = ['BG', "clown", "nface", 'color'] 


# source code from visulaize.py
def display_instances_cust(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, imagecount=0):

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 0, -0)
    ax.set_xlim(-0, width + 0)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{}".format(label)
        #ax.text(x1, y1 + 8, caption,                               #if not showing text
        #        color='b', size=20, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:                  
            masked_image = apply_mask(masked_image, mask, color)   

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            #Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor='none')
            ax.add_patch(p)
            

    ax.imshow(masked_image.astype(np.uint8))
    plt.savefig(f"/home/ubuntu/FobiaPhilter/ActionFiles/PostMRCNN/{imagecount}.jpg", bbox_inches='tight', transparent = True, pad_inches=-0.5, 
                orientation= 'landscape')  #save output
    if auto_show:
        plt.show()

        
        
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
     
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image





# Begining the ClownDataset
class ClownDataset(Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "clown")
        self.add_class("dataset", 2, "nface")   
        self.add_class("dataset", 3, "color")   
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        for filename in listdir(images_dir):
            image_id = filename[:-4]
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0,1,2,3]) 

    def extract_boxes(self, filename):
        tree = ElementTree.parse(filename)
        root = tree.getroot()
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height
    
    

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if i == 0:                                    
                masks[row_s:row_e, col_s:col_e, i] = 1                  
                class_ids.append(self.class_names.index('clown'))      
            elif i ==1:                                         
                masks[row_s:row_e, col_s:col_e, i] = 2                 
                class_ids.append(self.class_names.index('nface'))     
            elif i ==2:                                         
                masks[row_s:row_e, col_s:col_e, i] = 3                 
                class_ids.append(self.class_names.index('color'))
                
        return masks, asarray(class_ids, dtype='int32')                
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    
class PredictionConfig(Config):
    NAME = "Clown_cfg"
    NUM_CLASSES = 1 + 3
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.01
    
class InferenceConfig(coco.CocoConfig):  
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.01
    
class InferenceConfigOrig(coco.CocoConfig):  
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
        
    
def check_for_overlap(rectangle_a, rectangle_b):
    if(rectangle_a[0]>rectangle_b[2] or rectangle_a[1]>rectangle_b[3]):
        a = 'n'
    elif(rectangle_a[3]<rectangle_b[1] or rectangle_a[2]<rectangle_b[0]):
        a = 'n'
    else:
        a = 'y'
    return a
        
    
def plot_predicted_new(dataset, model, model2, cfg, cfg2, class_names, class_names2, n_images): 
    for i in range(n_images):
        image = dataset.load_image(i)
        
        #clown model 
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)               
        yhat = model.detect(sample, verbose=1)[0]     
        r = yhat
        
                
        #coco model 
        scaled_image = mold_image(image, cfg2)
        sample = expand_dims(scaled_image, 0)               
        yhat2 = model2.detect(sample, verbose=1)[0]     
        r2 = yhat2
        
        
        #condition
        for k in range(r['masks'].shape[-1]):
            if class_names[r['class_ids'][k]] == 'clown':
                clownBox = r['rois'][k]
                
                for coco in range(r2['masks'].shape[-1]):
                    if class_names2[r2['class_ids'][coco]] == 'person':
                        try:
                            if check_for_overlap(r['rois'][k], r2['rois'][coco])=='y':
                                mask = r2['masks'][:, :, coco]
                                image[mask] = 200
                                
                            else:
                                pass
                        except:
                            pass
                    else:
                        pass
                    
            elif class_names[r['class_ids'][k]] != 'clown':
                pass
            else:
                pass


        
        display_instances_cust(image, r2['rois'], r2['masks'],  r2['class_ids'], class_names2, scores=False, imagecount=i,
        show_bbox=False, captions=False, show_mask=False) 
        

        
def plot_predicted_coco(dataset, model3, cfg3, class_names2, n_images, phobia): 
    for i in range(n_images):
        image = dataset.load_image(i)

        #coco model 
        scaled_image = mold_image(image, cfg3)
        sample = expand_dims(scaled_image, 0)               
        yhat2 = model3.detect(sample, verbose=1)[0]     
        r2 = yhat2

        #condition               
        for coco in range(r2['masks'].shape[-1]):
            if class_names2[r2['class_ids'][coco]] == phobia:
                mask = r2['masks'][:, :, coco]
                image[mask] = 200
            else:
                pass

        display_instances_cust(image, r2['rois'], r2['masks'],  r2['class_ids'], class_names2, scores=False, imagecount=i,
        show_bbox=False, captions=False, show_mask=False) 

        
def Modelmain():
    test_set = ClownDataset()
    test_set.load_dataset('/home/ubuntu/FobiaPhilter/ActionFiles/FramesFromVideo', is_train=False)
    test_set.prepare()
    cfg = PredictionConfig()  

    model_path = '/home/ubuntu/FobiaPhilter/ActionFiles/model/mask_rcnn_clown_cfg_0025.h5'
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    model.load_weights(model_path, by_name=True)


    cfg2 = InferenceConfig()
    cfg3 = InferenceConfigOrig()    
    weights_path = '/home/ubuntu/FobiaPhilter/ActionFiles/model/mask_rcnn_coco.h5'
    
    model2 = MaskRCNN(mode='inference', model_dir='./', config=cfg2)
    model2.load_weights(weights_path, by_name=True)
    
    model3 = MaskRCNN(mode='inference', model_dir='./', config=cfg3)
    model3.load_weights(weights_path, by_name=True)
            
    if phobia == 'clown': 
        plot_predicted_new(test_set, model, model2, cfg, cfg2, class_names, class_names2, n_images)
        
        #Export imageID vs. original Filename
        files = []
        for m in test_set.image_from_source_map:
            files.append(m)

        df = pd.DataFrame({'Original_files':files})
        df['Index_outputFile'] = df.index
        df['Original_files'] = df['Original_files'].str.replace('dataset.image','').astype('int64')
        df = df.sort_values(by=['Original_files'])
        df.to_csv('/home/ubuntu/FobiaPhilter/ActionFiles/TestSampleImageID.txt')
            
    elif phobia != 'clown':   
        plot_predicted_coco(test_set, model3, cfg3, class_names2, n_images,  phobia)
        
        #Export imageID vs. original Filename
        files = []
        for m in test_set.image_from_source_map:
            files.append(m)

        df = pd.DataFrame({'Original_files':files})
        df['Index_outputFile'] = df.index
        df['Original_files'] = df['Original_files'].str.replace('dataset.image','').astype('int64')
        df = df.sort_values(by=['Original_files'])
        df.to_csv('/home/ubuntu/FobiaPhilter/ActionFiles/TestSampleImageID.txt')
        
    else:
        pass 
    

# convert images to videos
def convertImageToVideo():
    pathIn= '/home/ubuntu/FobiaPhilter/ActionFiles/PostMRCNN/'
    pathOut = '/home/ubuntu/FobiaPhilter/ActionFiles/videoConstruct1.mp4'
    df_filename_imageID = pd.read_csv('/home/ubuntu/FobiaPhilter/ActionFiles/TestSampleImageID.txt')
    
    frame_array = []
    for file in df_filename_imageID['Index_outputFile']:
        filename = pathIn + str(file) +'.jpg'
        try:
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            frame_array.append(img)
        except:
            pass
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()
    return out
    
    
def combine_audio(vidname, audname, outname, fps=30):
    import moviepy.editor as mpe
    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname,fps=fps)

    
#############################################

def main():
    st.write('Editing your request ...   A link will be sent to your email address')  
    download()
    
    sec = sec1 
    count=1
    
    
    # video - to -images 
    def getFrame(sec):
        vidcap = cv2.VideoCapture(bestResolutionVideo.filename)
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)   #set the capturing start at (sec*1000 milliseconds)
        hasFrames,image = vidcap.read()
        if hasFrames:
            SavePath = '/home/ubuntu/FobiaPhilter/ActionFiles/FramesFromVideo/images/'
            cv2.imwrite(SavePath + "image"+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
        
    
    success = getFrame(sec)
    while success:
        while (count < MaxCount):
            count = count + 1   
            sec = sec + (1/fps)*1    #every 1 frames
            sec = round(sec, 2)
            success = getFrame(sec)
        else:
            break
            
    audio(sec, MaxCount)        
    Modelmain()
    convertImageToVideo()
    combine_audio('/home/ubuntu/FobiaPhilter/ActionFiles/videoConstruct1.mp4', 'audio.mp3', f'{bestResolutionVideo.filename}')
    
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(f'/home/ubuntu/FobiaPhilter/{bestResolutionVideo.filename}', 
                               'vidobject', f'FearReFactor/{bestResolutionVideo.filename}')
    

    
    return
    
    
################################################################################################################################
###################################################################################################################################
    
st.markdown("### 🎲 Proceed?")


    
email = st.text_input('Email:',"", key = 'emailTo')

def emailResults(TO):
    # headers
    FROM = 'dr.hehannah@gmail.com'
    
    URL = f'https://vidobject.s3-us-west-2.amazonaws.com/FearReFactor/{bestResolutionVideo.filename}'
    message = 'Subject: {}\n\n{}'.format('Fear ReFactor', f'Your request is ready. Click to {URL}')
    
    
    # SMTP
    send = smtplib.SMTP('smtp.gmail.com', 587)
    send.starttls()
    send.login() #need to provide
    send.sendmail(FROM, TO, message)
    send.quit()
    

if st.button('Run & Share'):
    main()
    emailResults(email)

    
    
# To play video (! if running in aws, there won't be audio!)
def playVideo():    
    video_path = '/home/ubuntu/FobiaPhilter/ActionFiles/videoConstruct1.mp4' 
    audio_path = "/home/ubuntu/FobiaPhilter/audio.mp3"
    video = cv2.VideoCapture(video_path)
    player = MediaPlayer(audio_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()
    
       
#if st.button('PLAY'):
#    playVideo()


    
def playDemo(path):    
    video_path = path + 'videoConstruct1.mp4' 
    audio_path = path + "audio.mp3"
    video = cv2.VideoCapture(video_path)
    player = MediaPlayer(audio_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()
    

#if st.button('PLAY Demos?'):
#    original_path = '/home/ubuntu/FobiaPhilter/ActionFiles/Demo'
#    selected_path = 'Best Clown Pranks Compilation 2018 (Clown)'
#    final_path = original_path + "/" + selected_path + "/"
#    playDemo(final_path)
    
    

