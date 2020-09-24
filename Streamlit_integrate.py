import streamlit as st
from youtubesearchpython import SearchVideos
import pandas as pd
from lxml import html, etree
import pafy
from youtube_transcript_api import YouTubeTranscriptApi
import json
import os
import sys 
import cv2
from os.path import isfile, join

st.header('YouTube')
maxi = st.selectbox('Maximum Searches', range(0, 20), 1)
query = st.text_input("Search", "")

def YouTubeQuery(query:str, maxi:int, kind):
    search = SearchVideos(query, offset = 1, mode = "json", max_results = maxi)
    res = pd.DataFrame()
    res['Titles'] = search.titles
    res['link'] = search.links
    res['id'] = search.ids
    res['durations'] = search.durations
    res.to_csv('searchRes_{types}.txt'.format(types = kind))
    return res 

def downloadVideos():
    res = YouTubeQuery(query, maxi, "Videos")
    for url in res["link"]:
        video = pafy.new(url)
        bestResolutionVideo = video.getbest()
        bestResolutionVideo.download()  
            
def downloadIndTranscriptsVideos():
    res = YouTubeQuery(query, maxi, "Transcripts")
        
    for i in range(0,len(res)):
        try:
            trans = YouTubeTranscriptApi.get_transcript(res['id'].iloc[i], languages=['en'])
            with open('{title}_transcript.json'.format(title = res['Titles'][i]),'w') as json_file:   #this may need to polish json later on
                json.dump(trans, json_file)
            
            video = pafy.new(res['link'][i])
            bestResolutionVideo = video.getbest()
            bestResolutionVideo.download()                
        except: 
            print('Transcript no found',f"{res['Titles'].iloc[i]}")

    
def displayAllResults():
    df = pd.read_csv('searchRes_Videos.txt')
    df = df[['Titles']]
    st.write(df)
    return(df)
    
    
def displayOnlyPreruns():
    try:
        f = os.listdir(os.getcwd() + r"\Buffered")
        ResList = pd.read_csv('searchRes_Videos.txt')[['Titles']]
        ResList['mTitles'] = ResList['Titles']+".mp4"
        display = ResList[ResList['mTitles'].isin(f)][['Titles']]
        st.write(display)
    except:
        pass 
        
    
    
def playVideo():
    
    os.chdir(r'C:\Users\DrHeh\Google Drive\programming\py_InsightFellowProject\FobiaPhilter(scripts drafted local)\Buffered_converted_videos')
    cap = cv2.VideoCapture('IT ( Stephen King Trailer - 1990 ).mp4')
    while True:
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('frame',frame)
            cv2.waitKey(60)

            if cv2.waitKey(60) & 0xFF == ord('q'):   # Press Q on keyboard to  exit 
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    
    
    
    
        
def main():
    downloadVideos()
    #downloadIndTranscriptsVideos()
    st.header('Found')
    displayAllResults()
    if st.checkbox("See pre-processed"):
        displayOnlyPreruns()
        if st.checkbox("Start play"):
            playVideo()
            


      
if __name__ == "__main__":
    st.text('Now searching ...' + " " + query + ", with maximum" + ' ' + '%s' % maxi)
    main()

st.text('You are done!')