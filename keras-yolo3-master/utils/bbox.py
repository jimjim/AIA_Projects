import numpy as np
import os
import cv2
from .colors import get_color

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score      

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3    

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def draw_boxes(image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
                
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(label))
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(box.xmin+13, box.ymin - 13), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1e-3 * image.shape[0], 
                        color=(0,0,0), 
                        thickness=2)
        
    return image          
def _correct_box(s, e):
    if s <0:s = 0
    if e <0:e = 0
    if s>=e:
        print("incorrect box val %d:%d"%(s,e))
        s = e = 0
    return s,e

def draw_boxes_w_classifier(f_idx, cf, image, boxes, labels, obj_thresh, quiet=True):
    image_ori = image.copy()
    #box_info = dict() #{idx:[id, (p)]}
    box_info = {0:[], 1:[],2:[],3:[],4:[]}
    idx = 0

    for box in boxes:
        label_str = ''
        label = -1
        idx = idx + 1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
                
        if label >= 0:
            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]
            region = np.array([[box.xmin-3,        box.ymin], 
                               [box.xmin-3,        box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin-height-26], 
                               [box.xmin+width+13, box.ymin]], dtype='int32')  
            ### Test
            y0 , y1 = _correct_box(box.ymin, box.ymax)
            x0 , x1 = _correct_box(box.xmin, box.xmax)
            h = y1 - y0
            w = x1 - x0
            if h > w:
                w = h
            else:
                h = w
            frame_cropped_box = image_ori[y0:y0+h, x0:x0+w]
            #pid = cf.predict_Beetle_id(frame_cropped_box)
            pid,p = cf.predict_Beetle_id_wP(frame_cropped_box)
            box_info[pid].append([idx,p])
            label_str = "%d_%d_%s"%(idx, pid, label_str)
            #print(">>>detected %d"%pid)
            #img_path_cropped = os.path.join(out_img_folder, "%s_box_%d.jpg"%(_build_filename(lname,fcnt,f_idx), bug_id))   
            #img_path_cropped = './output/box_%d_%d.jpg'%(x0,y0)        
            #cv2.imwrite(img_path_cropped, frame_cropped_box)              
            ####

            cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            cv2.fillPoly(img=image, pts=[region], color=get_color(pid))
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(box.xmin+13, box.ymin - 13), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1e-3 * image.shape[0], 
                        color=(0,0,0), 
                        thickness=2)

    out_img_path = "./output/f%d.jpg"%f_idx
    cv2.imwrite(out_img_path, image)
    for k in box_info.keys():
        v = box_info[k]
        if len(v)>1:
            out_txt_path = "./output/f%d.log"%f_idx
            fw = open(out_txt_path,'w')
            fw.write(str(box_info))
            fw.close()
            print("Got error frame idx:%d"%idx)
            break


    return image 

def fprint(fw, ostr):
    if fw:
        fw.write(ostr + '\n')

def insert_box_info(boxs, boxs_xy, cid, bid, p_info, fw=None, tracked=True):
    new_score = p_info[cid]
    fprint(fw, "box:%d c:%d score:%04f"%(bid,cid,new_score))
    if new_score >=1:
        fprint(fw, "incorrect score %d-c%d score:%d"%(bid,cid,new_score))
    
    if boxs[cid][0] == 0: #no stroed score
        boxs[cid][0] = new_score
        boxs[cid].append([bid, p_info])
        if boxs_xy[bid]['cid']!= cid:
            print('ERROR! %d-%d-%d'%(bid, cid, boxs_xy[bid]['cid']))
        fprint(fw,"[A] Insert new info wo conflict c:%d"%cid)
        #fprint(fw,boxs)
        return cid, p_info
    else:
        if tracked and new_score>boxs[cid][0]: #replace boxes info
            fprint(fw, "[B] conflict 1")
            revised_info = list(boxs[cid][1][1])
            revised_info[cid] = -1
            revised_info = tuple(revised_info)
            restore_bid = boxs[cid][1][0]
            revised_cid = np.argmax(revised_info)
            boxs_xy[restore_bid]['cid'] = revised_cid
            boxs_xy[restore_bid]['revised'] = True
            fprint(fw, "unsorted rc%s-c%s-%s"%(revised_cid, restore_bid, revised_info))
            boxs[cid][0] = new_score
            boxs[cid][1][0] = bid
            boxs[cid][1][1] = p_info
            #print(boxs)
            return insert_box_info(boxs,boxs_xy,revised_cid, restore_bid,revised_info, fw)
        elif new_score <0:
            boxs_xy[bid]['revised'] = -1 #dont draw this box
            fprint(fw, "[D] box slots are full, discard this one bid%d"%bid)
            print("[D] box slots are full, discard this one bid%d"%bid)
            return cid, p_info
        else:    
            fprint(fw, "[C]conflict 2")
            revised_info = list(p_info)
            revised_info[cid] = -1    
            revised_info = tuple(revised_info)
            revised_cid = np.argmax(revised_info)
            boxs_xy[bid]['cid'] = revised_cid
            boxs_xy[bid]['revised'] = True

            fprint(fw, "revise c:%s-%s"%(revised_cid, revised_info))
            return insert_box_info(boxs, boxs_xy,revised_cid,bid, revised_info, fw)      


def insert_box_info_pure(boxs, cid, bid, p_info):
    pass

def draw_boxes_w_classifier_ex(f_idx, cf, image, boxes, labels, obj_thresh, quiet=True):
    image_ori = image.copy()
    #box_info = dict() #{idx:[id, (p)]}
    box_info = {0:[0], 1:[0],2:[0],3:[0],4:[0]}
    box_xy = dict()# {box_idx: [revised, box]}
    idx = 0

    out_txt_path = "./output/f%d.log"%f_idx
    fw = open(out_txt_path,'w')

    for box in boxes:
        #label_str = ''
        label = -1
        idx = idx + 1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label = i

        if label >= 0:
            y0 , y1 = _correct_box(box.ymin, box.ymax)
            x0 , x1 = _correct_box(box.xmin, box.xmax)
            h = y1 - y0
            w = x1 - x0
            if h > w:
                w = h
            else:
                h = w
            frame_cropped_box = image_ori[y0:y0+h, x0:x0+w]
            cid,p = cf.predict_Beetle_id_wP(frame_cropped_box)
            
            box_xy[idx] = {'revised':0, 'box':box, 'label':label, 'cid':cid}
            insert_box_info(box_info, box_xy, cid, idx, p, fw)
            #box_info[cid].append([idx,p])

    fw.write(str(box_info))
    fw.write('\n')
    fw.write(str(box_xy))
    ### Draw all boxes test
    revised_img = False
    for bid in box_xy.keys():
        box = box_xy[bid]['box']
        label = box_xy[bid]['label']
        revised = box_xy[bid]['revised']
        if revised<0:
            print("f:%d drop conflict boxes"%f_idx)
            continue
        cid = box_xy[bid]['cid']

        label_str = (labels[label] + ' ' + str(round(box.get_score()*100, 2)) + '%')
        if revised:
            label_str = "%d_r%d_%s"%(bid, cid, label_str)
            cr = (0,0,255)
            print("f:%d revise box%d"%(f_idx, bid))
            revised_img = True
        else:
            label_str = "%d_c%d_%s"%(bid, cid, label_str)
            cr = get_color(cid)

        text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
        width, height = text_size[0][0], text_size[0][1]
        region = np.array([[box.xmin-3,        box.ymin], 
                            [box.xmin-3,        box.ymin-height-26], 
                            [box.xmin+width+13, box.ymin-height-26], 
                            [box.xmin+width+13, box.ymin]], dtype='int32')    

        cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=cr, thickness=2)
        cv2.fillPoly(img=image, pts=[region], color=get_color(cid))
        cv2.putText(img=image, 
                    text=label_str, 
                    org=(box.xmin+13, box.ymin - 13), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1e-3 * image.shape[0], 
                    color=(0,0,0), 
                    thickness=2)                            
    
    if not revised_img:
        out_img_path = "./output/f%d.jpg"%f_idx
    else:
        out_img_path = "./output/revised_f%d.jpg"%f_idx   

    cv2.imwrite(out_img_path, image)
    fw.close()
    ''' 
    for k in box_info.keys():
        v = box_info[k]
        if len(v)>1:
            out_txt_path = "./output/f%d.log"%f_idx
            fw = open(out_txt_path,'w')
            fw.write(str(box_info))
            fw.close()
            print("Got error frame idx:%d"%f_idx)
            break
    '''
    return image 

def draw_boxes_w_classifier_sort(f_idx, cf, sort, track_id_map, image, boxes, labels,label_img_list, obj_thresh, quiet=True):
    image_ori = image.copy()
    box_info = {0:[0], 1:[0],2:[0],3:[0],4:[0]}
    frame_det = list()
    box_xy = dict()# {box_idx: [revised, box]}
    idx = 0

    out_txt_path = "./output/f%d.log"%f_idx
    fw = open(out_txt_path,'w')

    for box in boxes:
        #label_str = ''
        label = -1
        idx = idx + 1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label = i

        if label >= 0:
            y0 , y1 = _correct_box(box.ymin, box.ymax)
            x0 , x1 = _correct_box(box.xmin, box.xmax)
            h = y1 - y0
            w = x1 - x0
            if h > w:
                w = h
            else:
                h = w
            #frame_cropped_box = image_ori[y0:y0+h, x0:x0+w]
            #cid,p = cf.predict_Beetle_id_wP(frame_cropped_box)
            #cid = 0
            #box_xy[idx] = {'revised':0, 'box':box, 'label':label, 'cid':cid}
            frame_det.append([x0,y0,x0+w,y0+h, box.get_score()])
            #insert_box_info(box_info, box_xy, cid, idx, p, fw)
            #box_info[cid].append([idx,p])



    frame_det = np.array(frame_det)
    #print(frame_det)
    track_bbs_ids = sort.update(frame_det, f_idx)

    track_bbs_ids = track_bbs_ids[track_bbs_ids[:,4].argsort()]
    
    for r in track_bbs_ids:
        #print(r)
        track_id = int(r[4])
        box = BoundBox(int(r[0]),int(r[1]),int(r[2]),int(r[3]))
        
        tracked = False
        #print(box.xmin, box.ymin, box.xmax, box.ymax)
        if track_id in track_id_map.keys():
            cid = track_id_map[track_id][0]
            p = track_id_map[track_id][1]
            tracked = True
        else:
            #new track id
            y0 , y1 = _correct_box(box.ymin, box.ymax)
            x0 , x1 = _correct_box(box.xmin, box.xmax)
            h = y1 - y0
            w = x1 - x0
            if h > w:
                w = h
            else:
                h = w
            frame_cropped_box = image_ori[y0:y0+h, x0:x0+w]            
            cid,p = cf.predict_Beetle_id_wP(frame_cropped_box)
            if p[cid] < 0.8:
                print("f:%d ignore low accuracy box t:%d c:%d p:%03f"%(f_idx, track_id , cid, p[cid]))
                continue
            if p[cid] >= 1.0:
                print("f:%d ignore abnormal box t:%d c:%d p:%s"%(f_idx, track_id, cid, str(p)))
                continue
                
            track_id_map[track_id] = [cid, p]

        labels = ['Moomin', 'Snork', 'Mymble','Sniff']
        box_xy[track_id] = {'revised':0, 'box':box, 'labels':labels, 'cid':cid}
        cid, p = insert_box_info(box_info, box_xy, cid, track_id, p, fw, tracked)
        track_id_map[track_id] = [cid, p]

    fw.write("\nframe_det:\n")
    fw.write(str(frame_det))
    fw.write("\ntrackers:\n")
    fw.write(str(track_bbs_ids))
    fw.write("\nbox_xy\n")
    fw.write(str(box_xy))
    fw.write("\ntrack_id_map\n")
    fw.write(str(track_id_map))
    fw.write("\ntrack_ids\n")
    fw.write(str(sort.get_trackers_id()))
        
    ### Draw all boxes test
    revised_img = False
    for bid in box_xy.keys():
        box = box_xy[bid]['box']
        
        revised = box_xy[bid]['revised']
        if revised<0:
            print("f:%d drop conflict boxes"%f_idx)
            continue
        cid = box_xy[bid]['cid']   
        label = box_xy[bid]['labels'][cid]
        #print(f_idx, cid, box.xmin, box.xmax, box.ymin, box.ymax)
        label_str = label
        if revised:
            label_str = "   %s (tkr:%d)"%( label_str,bid)
            cr = (0,0,255)
            #print("f:%d revise box%d"%(f_idx, bid))
            revised_img = True
        else:
            label_str = "   %s (tkr:%d)"%(label_str,bid)
            cr = get_color(cid)

        s_img = label_img_list[cid]
        x_offset = box.xmin +1
        y_offset = box.ymin -40
        
        text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
        width, height = text_size[0][0], text_size[0][1]
        region = np.array([[box.xmin-3,        box.ymin], 
                            [box.xmin-3,        box.ymin-height-26], 
                            [box.xmin+width+13, box.ymin-height-26], 
                            [box.xmin+width+13, box.ymin]], dtype='int32')    

        cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=cr, thickness=2)
        cv2.fillPoly(img=image, pts=[region], color=get_color(cid))
        cv2.putText(img=image, 
                    text=label_str, 
                    org=(box.xmin+13, box.ymin - 13), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1e-3 * image.shape[0], 
                    color=(0,0,0), 
                    thickness=2) 
        if x_offset+s_img.shape[1]<image.shape[1] and y_offset+s_img.shape[0]<image.shape[0]:
            image[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

    if not revised_img:
        out_img_path = "./output/f%d.jpg"%f_idx
    else:
        out_img_path = "./output/revised_f%d.jpg"%f_idx   
    cv2.putText(img=image, 
                text="%04d"%f_idx, 
                org=(13, 20), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1e-3 * image.shape[0], 
                color=(255,255,200), 
                thickness=2)   

    #cv2.imwrite(out_img_path, image)
    fw.close()
    return image 