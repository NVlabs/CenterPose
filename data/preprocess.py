# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial.
# Full text can be found in LICENSE.md

import os
import glob
import argparse
import cv2
import tqdm
from utils import *

import subprocess

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from objectron.schema import annotation_data_pb2 as annotation_protocol

categories = [
"bike",
# "book",
# "bottle",
# "camera",
# "cereal_box",
# "chair",
# "cup",
# "laptop",
# "shoe"
]

def grab_frame(video_file, sequence,frame_ids):
    """Grab an image frame from the video file."""

    try:
        frames = []

        # Numpy array width (img height) <- data.camera.image_resolution_width
        width,height = sequence.frame_annotations[0].camera.image_resolution_width,sequence.frame_annotations[0].camera.image_resolution_height

        # Debug only
        # frame_ids=frame_ids[0:310]

        frame_size = len(frame_ids)*width * height * 3
        
        frame_filter='select=\''
        for idx,frame_id in enumerate(frame_ids):
            if idx==0:
                frame_filter = frame_filter+f'eq(n\,{frame_id})'
            else:
                frame_filter = frame_filter+f'+eq(n\,{frame_id})'

        frame_filter=frame_filter+'\''
        command = [
            'ffmpeg', '-i', video_file, '-f', 'image2pipe', '-vf', frame_filter,
            '-pix_fmt', 'rgb24', '-vcodec', 'rawvideo', '-vsync', 'vfr', '-','-loglevel', 'panic','-hide_banner'
        ]
        # pipe = subprocess.Popen(
        #     command, stdout=subprocess.PIPE, bufsize = 2* frame_size)
        pipe = subprocess.Popen(
            command, stdout=subprocess.PIPE)
        current_frame=np.frombuffer(
            pipe.stdout.read(frame_size), dtype='uint8')

        if current_frame.size==frame_size:

            current_frame = current_frame.reshape(int(current_frame.size/width/height/3),width, height, 3)
            pipe.stdout.flush()
            if frames == []:
                frames=current_frame
            else:
                frames=frames+current_frame

            warning_flag=0
        else:
            warning_flag=1
    except:
        return None, 1

    return frames,warning_flag

def preprocess(annotation_file,category,opt):
    # Read from the sequence info

    try:
        with open(annotation_file, 'rb') as pb:
            sequence = annotation_protocol.Sequence()
            sequence.ParseFromString(pb.read())
    except:
        with open('bug_list.txt', 'a+') as fp:
            video_filename = annotation_file.replace('pbdata','MOV')
            fp.write(video_filename)
            fp.write('\n')
        return


    frame_id_list=list(range(0,len(sequence.frame_annotations),opt.frame_rate))
    
    # Debug only
    # frame_id_list=frame_id_list[0:300]

    # Extract all the frames from the video
    video_filename=annotation_file.replace('pbdata','MOV')
    frame, warning_flag = grab_frame(video_filename, sequence,frame_id_list)

    if warning_flag==1:
        
        with open('bug_list.txt', 'a+') as fp:
            fp.write(video_filename)
            fp.write('\n')
    else:
        prefix=annotation_file[annotation_file.rfind('/')+1:annotation_file.rfind('.')]
        if os.path.isdir(f'{opt.outf}/{category}/{prefix}'):
            print(f'folder {opt.outf}/{category}/{prefix} exists')
        else:
            os.mkdir(f'{opt.outf}/{category}/{prefix}')
            print(f'created folder {opt.outf}/{category}/{prefix}')

        for i,frame_id in enumerate(frame_id_list):
            # Debug only
            # print(f"{str(frame_id).zfill(5)}")

            # Save all the extracted images
            im_bgr = cv2.cvtColor(frame[i], cv2.COLOR_RGB2BGR)
            
            # resize to img width, img height
            im_bgr=cv2.resize(im_bgr, (int(im_bgr.shape[1]/opt.resolution_ratio),int(im_bgr.shape[0]/opt.resolution_ratio)))

            cv2.imwrite(f"{opt.outf}/{category}/{prefix}/{str(frame_id).zfill(5)}.png",im_bgr)

            # Export .json file
            warning_flag = export_to_ndds_file(
                frame,
                f"{opt.outf}/{category}/{prefix}/{str(frame_id).zfill(5)}.json",
                # f"{opt.outf}/{category}/{prefix}.json",
                sequence=sequence,
                frame_id=frame_id,
                opt=opt,
                video_filename=video_filename
            )

            if warning_flag == 1:
                with open('bug_list.txt', 'a+') as fp:
                    fp.write(video_filename)
                    fp.write('\n')
                return
           

if __name__ == "__main__":

    # User defined parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--outf',
        default = 'outf_all',
        help = "output filename inside output/"
    )
    parser.add_argument(
        '--debug',
        action = 'store_true',
        default = False,
        help = "Debug mode or not"
    )
    parser.add_argument(
        '--resolution_ratio',
        default = 2.4,
        help = "ratio change from the original resolution (by default 1920*1440-> 800*600)"
    )
    parser.add_argument(
        '--frame_rate',
        type = int,
        default = 1,
        help = "skip intermediate frames"
    )
    parser.add_argument(
        '--skip',
        action = 'store_true',
        default = False,
        help = "skip the files which have been generated"
    )
    parser.add_argument(
        '--test_flag',
        action = 'store_true',
        default = False,
        help = "generate data for test"
    )
    parser.add_argument(
        '--c',
        nargs = '+',
        default = categories,
        help = "categories to be generated"
    )
    opt = parser.parse_args()

    # Todo: Hack some parameters here, should be commented if not using Pycharm but .sh instead
    # opt.debug = True
    # opt.test_flag = True
    opt.skip = True
    opt.resolution_ratio = 2.4

    if os.path.isdir(f'{opt.outf}'):
        print(f'folder {opt.outf}/ exists')
    else:
        os.mkdir(f'{opt.outf}')
        print(f'created folder {opt.outf}/')

    # Target file

    if opt.debug is True:
        
        annotation_file = 'test/chair_batch-13_32.pbdata'
    
        if os.path.isdir(f'{opt.outf}/debug'):
            print(f'folder {opt.outf}/debug exists')
        else:
            os.mkdir(f'{opt.outf}/debug')
            print(f'created folder {opt.outf}/debug')

        if opt.skip == True:
            prefix=annotation_file[annotation_file.rfind('/')+1:annotation_file.rfind('.')]
            if glob.glob(f"{opt.outf}/debug/{prefix}/*.json"):
                print('Skip it')
            else:
                preprocess(annotation_file,'debug',opt)
        else:
            preprocess(annotation_file,'debug',opt)
    else:
        for c in opt.c:
            print(c)
            if opt.test_flag == False:
                suffix="train"
            else:
                suffix="test"

            if os.path.isdir(f'{opt.outf}/{c}_{suffix}'):
                print(f'folder {opt.outf}/{c}_{suffix} exists')
            else:
                os.mkdir(f'{opt.outf}/{c}_{suffix}')
                print(f'created folder {opt.outf}/{c}_{suffix}')

            
            with open(f"index" + f"/{c}_annotations_{suffix}",'r') as fopen:
                target_list = fopen.read().splitlines()
            
            # Read bug list
            if os.path.exists(f"label/{c}/bug_list.txt"):
                with open(f"label/{c}/bug_list.txt",'r') as fopen:
                    opt.bug_list = fopen.read().splitlines()
            else:
                opt.bug_list = []


            for target in tqdm.tqdm(target_list):
                print(target)
                annotation_file = f"data/{c}/"+target.replace('/','_')+'.pbdata'
                
                
                prefix = annotation_file[annotation_file.rfind('/')+1:annotation_file.rfind('.')] 

                if opt.skip == True:
                    # if glob.glob(f"{opt.outf}/{c}_{suffix}/{prefix}.png"):
                    if glob.glob(f"{opt.outf}/{c}_{suffix}/{prefix}/*.json"):
                        continue

                                
                preprocess(annotation_file,f"{c}_{suffix}",opt)

        print('Done')
    

       
