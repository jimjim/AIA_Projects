import os
import argparse

def _main_(args):
    input_file   = args.input

    if not os.path.exists(input_file):
        print "not existed: %s"%input_file
        return

    dur = 30 #mins
    cmd = ""
    output_p = input_file[:-4]
    output_p = output_p.replace(" ","")
    for x in range(dur):
        output_file = output_p + "_m_%02d.avi"%x
        cmd = 'ffmpeg -ss 00:%02d:00 -t 60 -i "%s" -vcodec copy -acodec copy %s'%(x, input_file, output_file)
        print(cmd+"\n")
        os.system(cmd)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    
    args = argparser.parse_args()
    _main_(args)