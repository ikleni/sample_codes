import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm
import numpy as np
import pytesseract
import subprocess


# useful objects 
directory = '/Users/IvanK/NFL/VideoData'
os.chdir(directory)

pytesseract.pytesseract.tesseract_cmd = (r'/Users/IvanK/tesseract/tesseract')

tessdata_dir_config = r'--tessdata-dir "/Users/IvanK/tesseract/tessdata" --psm 10 --oem 3 -c tessedit_char_whitelist=0123456789' # try # --oem 0?
# kernel = np.ones((2,2), np.uint8)

# binarisation constant
threshold = 180
# sizes and colours for expanding an image
top, bottom = int(20), int(20)
left, right = int(35), int(35)
white, black  = [255,255,255], [0,0,0]

def keywithmaxval(d):
    """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(max(v))]

def basic_image_preproc(im_path, threshold = 180):
    # make sure that the path is correct
    tmp_img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

    # we need to convert from BGR to RGB format/mode:
    img_rgb = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
    img_grey =  cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)

    _, otsu_s = cv2.threshold(img_grey,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, im_white = cv2.threshold(img_rgb,threshold,255,cv2.THRESH_BINARY_INV)
    
    # make the size uniform?
    otsu_s = cv2.copyMakeBorder(otsu_s,top,bottom,left,right, cv2.BORDER_CONSTANT, value = white)
    im_white = cv2.copyMakeBorder(im_white,top,bottom,left,right, cv2.BORDER_CONSTANT, value = white)

    return otsu_s, im_white



##############################################
##############################################
##############################################
############## Video to images ###############
##############################################
##############################################
##############################################



def video_to_images(v_name, max_frames = 2000):

    cap= cv2.VideoCapture(v_name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    freq = total // max_frames

    i = 0 # for naming 
    n = 0 # iterator

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if (n % freq == 0):
            cv2.imwrite(f'tmp/frames/F{i}.jpg',frame)
            i+=1
        n += 1
        
    cap.release()
    cv2.destroyAllWindows()
    
    
    
##############################################
##############################################
##############################################
############# Detecting Channel ##############
##############################################
##############################################
##############################################
'''does not work on this data'''



# retrieve keys 
fox = cv2.imread('KeyImages/fox.jpg', cv2.IMREAD_UNCHANGED)
nbc = cv2.imread('KeyImages/nbc.jpg', cv2.IMREAD_UNCHANGED)
cbs = cv2.imread('KeyImages/cbs.jpg', cv2.IMREAD_UNCHANGED)
espn = cv2.imread('KeyImages/espn.jpg', cv2.IMREAD_UNCHANGED)


channels_keys = {'fox':fox, 'nbc':nbc, 'cbs':cbs, 'espn': espn}

def match_patterns(sample_img_names, video_id, keys):

    # allow for multiple key images per channel
    
    matching_results = {}

    for key_name, k in keys.items():

        tmp_placeholder = []
        
        for im_name in sample_img_names:
            
            im_name = f'Videos/{video_id}/frames/{im_name}'
            
            tmp_img = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
#             imageGray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
#             templateGray = cv2.cvtColor(k, cv2.COLOR_BGR2GRAY)

#             print(tmp_img.shape)
#             print(key_name)
#             print(k.shape)
            tmp_result = cv2.matchTemplate(tmp_img,k, cv2.TM_CCOEFF_NORMED)
            
            _, max_val, _, _ = cv2.minMaxLoc(tmp_result)
            tmp_placeholder.append(max_val)
                
            cv2.destroyAllWindows()
        tmp_max_val = np.median(tmp_placeholder)
        matching_results[key_name] = tmp_max_val
    
    print(matching_results)
    best_key = keywithmaxval(matching_results)

    return best_key

def get_matched_channel_for_video(v_id, S = 100):

    directory = f'Videos/{v_id}/frames'
    files = sorted(os.listdir(directory))
    N_files = len(files)

    files_sample = files[N_files//2 - S: N_files//2 + S]

    res = match_patterns(files_sample, v_id, channels_keys)
    # save results to df with video/matched with channel
    if 'video_channel_matched.csv' in os.listdir():
        df = pd.read_csv('video_channel_matched.csv')
        if v_id in list(df['video_id'].unique()):
            df.loc[df['video_id'] == v_id, 'channel_name'] =  res
            df.to_csv('video_channel_matched.csv', index = False)
            
        else:
            print('here')
            df = df.append({'video_id': v_id, 'channel_name':res}, ignore_index = True)
            print(df.head())
            df.to_csv('video_channel_matched.csv', index = False)

    else:
        df = pd.DataFrame(data = {'video_id': [v_id] ,'channel_name':  [res]})
        df.to_csv('video_channel_matched.csv', index = False)    

        
        
##############################################
##############################################
##############################################
############## Finding Score Imgs ############
##############################################
##############################################
##############################################



channel_box_locations_d = {'fox': [(572,550), (572,643), [(51,87), (51,85)]], 'nbc': [(630,428), (630,682), [(50,64), (50,58)]],
                           'cbs': [(615,430), (615,618), [(45,53), (45,57)]], 'espn': [(640,405), (640,680), [(54,130),(54,130)]]}

def get_bounding_boxs_wth_scores(channel_name):


    team_A_loc, team_B_loc, box_size = channel_box_locations_d[channel_name][0], channel_box_locations_d[channel_name][1], channel_box_locations_d[channel_name][2]
    h_a, w_a  = box_size[0][0], box_size[0][1]
    h_b, w_b  = box_size[1][0], box_size[1][1]

    directory = f'tmp/frames'
    # get files and sort by date
    files = os.listdir(directory)
    files.sort(key=lambda s: os.path.getmtime(os.path.join(directory, s)))
    
    print(files)
    # DS STORE ISSUE
    N = len(files)

    for n, name in enumerate(tqdm(files)):
        
        im_name = f'tmp/frames/{name}'
        
        tmp_img = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)

        score_A_img = tmp_img[team_A_loc[0]: team_A_loc[0] +h_a, team_A_loc[1]: team_A_loc[1] +w_a,:]
                                    
        score_B_img = tmp_img[team_B_loc[0]: team_B_loc[0] +h_b, team_B_loc[1]: team_B_loc[1] +w_b,:]
        
        cv2.imwrite(f'tmp/scoreboxes/F{n}_teamA.jpg',score_A_img)
        cv2.imwrite(f'tmp/scoreboxes/F{n}_teamB.jpg',score_B_img)
    
    
    cv2.destroyAllWindows()

    
    
##############################################
##############################################
##############################################
########## Get Score Data and PostProc #######
##############################################
##############################################
##############################################
    
    
    
def get_scores_for_video():

    directory = f'tmp/scoreboxes'

    # get files and sort by date (assuming ordered by time and team)
    files = os.listdir(directory)
    files.sort(key=lambda s: os.path.getmtime(os.path.join(directory, s)))
    files_A = files[1::2]
    files_B = files[2::2]

    # initialise score placeholders
    scores_A, scores_B = [],[]

    for n, name in enumerate(tqdm(files_A)):
        im_name = f'tmp/scoreboxes/{name}'
        blackened, whitened = basic_image_preproc(im_name)  

        score_black = pytesseract.image_to_string(blackened, config=tessdata_dir_config)
        score_white = pytesseract.image_to_string(whitened, config=tessdata_dir_config)
        
        score_joint = score_black + '&' + score_white
        
#         if (score_black == score_white):
#             scores_A.append(score_black)
#         else:
#             scores_A.append(np.nan)
            
        scores_A.append(score_joint)

    for n, name in enumerate(tqdm(files_B)):
        im_name = f'tmp/scoreboxes/{name}'
        blackened, whitened = basic_image_preproc(im_name)  

        score_black = pytesseract.image_to_string(blackened, config=tessdata_dir_config)
        score_white = pytesseract.image_to_string(whitened, config=tessdata_dir_config)
        
        score_joint = score_black + '&' + score_white
        
#         if (score_black == score_white):
#             scores_B.append(score_black)
#         else:
#             scores_B.append(np.nan)
        scores_B.append(score_joint)
    
    return scores_A, scores_B



##############################################
##############################################
##############################################
################## Big Loop ##################
##############################################
##############################################
##############################################



main_directory = '/Users/IvanK/NFL/VideoData'
video_table = pd.read_csv('merged_match_views_with_channel_2021.csv')
folder_paths = ['tmp/frames','tmp/scoreboxes']
n = 0 

for index, v in video_table.iterrows():
    
    n += 1
    
    os.chdir(main_directory) # return to the main one in case we have moved
    
    v_url = 'https://www.youtube.com/' + v['videoId']
    v_channel = v['channel']
    
    v_id = v['videoId']
        
    extracted_videos = os.listdir('/Users/IvanK/NFL/VideoData/scores')    
    v_in_extracted = f'{v_id}.csv' in extracted_videos
    
    if (v_channel == 'fox')|(v_in_extracted):
        pass
    
    else:

        subprocess.run(["pytube", v_url])

        files = os.listdir(directory)
        files.sort(key=lambda s: os.path.getmtime(os.path.join(directory, s)))
        print(files)
        v_name = files[-1]
        print(v_name)

        print('Converting to images!')
        video_to_images(v_name) # naming issue
        print('Getting bounding boxes!')
        get_bounding_boxs_wth_scores(v_channel)
        print('Getting scores boxes!')
        scores_A, scores_B = get_scores_for_video()

        # delete part
        os.unlink(v_name) # delete video

        for folder_path in folder_paths: # detele generated images
            for file_object in os.listdir(folder_path):
                file_object_path = os.path.join(folder_path, file_object)
                if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
                    os.unlink(file_object_path)

        # post processing part
        # make score files the same length since +- 1 image can happen when cropping
        n_min = min(len(scores_A), len(scores_B))

        if len(scores_A) > n_min:
            scores_A = scores_A[:n_min]
        elif len(scores_B) > n_min:
            scores_B = scores_B[:n_min]

        scores_df = pd.DataFrame({'Score_A': scores_A, 'Score_B' :scores_B})
        scores_df.to_csv(f'scores/{v_id}.csv', index = False)
    #     scores_A_processed, scores_B_processed = post_processing(scores_A), post_processing(scores_B)

    #     if n >=1:
    #         break


        # saving results
        # ???

        # example 3 team AB reversed 
    # if empty or something in one -> choose what the other says

    # figure out sorting/os and d.s store stuff
















