import numpy as np
import argparse
import sys
import os
import json
from scipy.spatial.transform import Rotation
from bbox3d import BBox3D
from metrics import iou_3d
import matplotlib.pyplot as plt
from numba import jit
import shutil
import pyquaternion


counter = 0 

def build_argo_object_list(label_data, classes_included = ['VEHICLE']):
	label_objects_list = []
	label_confidence_list = []
	for data in label_data:
		if data['label_class'] in classes_included:
			cx, cy, cz = data['center']['x'], data['center']['y'], data['center']['z']
			l, w, h = data['length'], data['width'], data['height']
			rw, rx, ry, rz = data['rotation']['w'], data['rotation']['x'],data['rotation']['y'],data['rotation']['z']

			temp = BBox3D(cx, cy, cz,
        	         length=l, width=w, height=h,
        	         rw=rw, rx=rx, ry=ry, rz=rz, is_center=True)
			label_objects_list.append(temp)
			if 'score' in data.keys():
				score = data['score']
				label_confidence_list.append(score)

	return label_objects_list,label_confidence_list


def valid_box_filter(box3d_list, score_ls=None):
	valid_gt_list = []
	valid_score_list = []

	for i,box3d in enumerate(box3d_list):
		if( (  (np.abs(box3d.cx) <= PC_AREA_SCOPE[2][3]) & (np.abs(box3d.cx) >= PC_AREA_SCOPE[2][2]) )\
		  & (  (np.abs(box3d.cy) <= PC_AREA_SCOPE[0][3]) & (np.abs(box3d.cy) >= PC_AREA_SCOPE[0][2]) ) \
		  & (  (np.abs(box3d.cz) <= PC_AREA_SCOPE[1][3]) & (np.abs(box3d.cz) >= PC_AREA_SCOPE[1][2]) ) ):
			valid_gt_list.append(box3d)
			if score_ls is not None:
				valid_score_list.append(score_ls[i])

	return valid_gt_list,valid_score_list

def evaluate_labels(prediction_data,prediction_score,label_data,label_score, IOU_threshold, lamda, label_files_name, output_name):

	if (IOU_threshold > 1) & (IOU_threshold < 0):
		raise('IOU Threshold should be between 0 and 1')
		
	
	hard_sample_maining = 0.0

	TP = 0.0
	valid_gt_list,label_score = valid_box_filter(label_data, label_score)
	prediction_data,prediction_score = valid_box_filter(prediction_data, prediction_score)
	total_gt = len(valid_gt_list)
	visited = [False for i in range(total_gt)]

	data = []	
	fname =	f"{output_name}/{label_files_name}"

	argo_output_file = fname
	
	# lambda
	#lamda = 0.1
	with open(argo_output_file,'w') as f:
		for j, predict_box in enumerate(prediction_data):
			TP = 0
			bbox3d = predict_box
			for i, label_box in enumerate(valid_gt_list):
				IOU_3D = iou_3d(predict_box, label_box)
				
				if(IOU_3D > IOU_threshold) and (visited[i] == False):
					visited[i] = True
					TP += 1.0
					
					
					if prediction_score[j] > label_score[i]:
						bbox3d = predict_box
						scores = prediction_score[j]
					else:
						bbox3d = label_box
						scores = label_score[i]

					temp = {}
					temp['center'] = {'x':float(bbox3d.cx), 'y':float(bbox3d.cy), 'z':float(bbox3d.cz)}
					temp['length'] = float(bbox3d.l)
					temp['width']  = float(bbox3d.w)
					temp['height'] = float(bbox3d.h)

					temp['rotation'] = {'w': bbox3d.q[0], 'x': bbox3d.q[1], 'y': bbox3d.q[2], 'z': bbox3d.q[3]}
					temp['score'] = min(float(scores) + lamda, 1.0)  # add weight to the confidence
					temp['label_class'] = 'VEHICLE'

					data.append(temp)

					
			if TP==0:
				bbox3d = predict_box
				scores = prediction_score[j]
				temp = {}
				temp['center'] = {'x':float(bbox3d.cx), 'y':float(bbox3d.cy), 'z':float(bbox3d.cz)}
				temp['length'] = float(bbox3d.l)
				temp['width']  = float(bbox3d.w)
				temp['height'] = float(bbox3d.h)
				temp['rotation'] = {'w': bbox3d.q[0], 'x': bbox3d.q[1], 'y': bbox3d.q[2], 'z': bbox3d.q[3]}
				temp['score'] = float(scores)
				temp['label_class'] = 'VEHICLE'
				data.append(temp)		
					
		for i, label_box in enumerate(valid_gt_list):
			if visited[i] == False:

				bbox3d = label_box
				scores = label_score[i]
				temp = {}
				temp['center'] = {'x':float(bbox3d.cx), 'y':float(bbox3d.cy), 'z':float(bbox3d.cz)}
				temp['length'] = float(bbox3d.l)
				temp['width']  = float(bbox3d.w)
				temp['height'] = float(bbox3d.h)
				temp['rotation'] = {'w': bbox3d.q[0], 'x': bbox3d.q[1], 'y': bbox3d.q[2], 'z': bbox3d.q[3]}
				temp['score'] = float(scores)
				temp['label_class'] = 'VEHICLE'
				data.append(temp)
				
		json.dump(data,f)


def single_iou_evaluation(prediction_path, label_path, output_name, iou=0.1, lamda=0.1):


	prediction_files = sorted(os.listdir(prediction_path))
	label_files = sorted(os.listdir(label_path))

	total_frames = len(prediction_files)	

	global counter


	for i in range(total_frames):
		prediction_content =  open(os.path.join(args.prediction_path,prediction_files[i]))
		label_content = open(os.path.join(args.label_path,label_files[i]))

		prediction_data, prediction_score = build_argo_object_list(json.load(prediction_content))
		label_data, label_score = build_argo_object_list(json.load(label_content))
		
		evaluate_labels(prediction_data, prediction_score, label_data, label_score, iou, lamda, label_files[i], output_name )


if __name__ == '__main__':	

	parser = argparse.ArgumentParser()
	parser.add_argument('--label_path', type = str,help = "Path of the labels")
	parser.add_argument('--prediction_path', type = str,help = "Path of the prediction")
	parser.add_argument('--iou', type =float, help = "IOU Threshold")
	parser.add_argument('--lamda', type =float, help = "lamda weight")	
	parser.add_argument('--pcmin', type =float, help = "pcmin Threshold")
	parser.add_argument('--pcmax', type =float, help = "pcmax Threshold")
	parser.add_argument('--output', type =str, help = "folder name")
	args = parser.parse_args()
	
	if ((args.pcmin == 0) & (args.pcmax == 100)):
	    PC_AREA_SCOPE = [[-100, 0, 0, 100], [-3, 0, 0,  3], [-100, 0, 0, 100]]
	elif ((args.pcmin == 0) & (args.pcmax == 80)):
	    PC_AREA_SCOPE = [[-80, 0, 0, 80],   [-3, 0, 0, 3], [-80, 0, 0, 80]]
	elif ((args.pcmin == 0) & (args.pcmax == 40)):
	    PC_AREA_SCOPE = [[-40, 0, 0, 40],   [-3, 0, 0, 3], [-40, 0, 0, 40]]
	elif ((args.pcmin == 0) & (args.pcmax == 20)):
	    PC_AREA_SCOPE = [[-20, 0, 0, 20],   [-3, 0, 0, 3], [-20, 0, 0, 20]]
	elif ((args.pcmin == 40) & (args.pcmax == 60)):
	    PC_AREA_SCOPE = [[-100, -0, 0, 100],   [-3, 0, 0, 3], [-60, -40, 40, 60]]
	elif ((args.pcmin == 60) & (args.pcmax == 80)):
	    PC_AREA_SCOPE = [[-100, -0, 0, 100],   [-3, 0, 0, 3], [-80, -60, 60, 80]]
	elif ((args.pcmin == 80) & (args.pcmax == 100)):
	    PC_AREA_SCOPE = [[-100, -0, 0, 100],   [-3, 0, 0, 3], [-100, -80, 80, 100]]	    
	
	if not os.path.isdir(args.label_path):
		print("label directory doesn't exist")

	if not os.path.isdir(args.prediction_path):
		print("prediction directory doesn't exist")
		
	
	print("Evaluation started \n **************************** ")
	print("***Evaluation on PC_lidar span of {} at iou threshold of {} is started***".format(PC_AREA_SCOPE, args.iou ))

	## Copy the labels of the reference frame 
	if not os.path.exists(f"{args.output}"):
		os.makedirs(f"{args.output}")	
	
	single_iou_evaluation(args.prediction_path, args.label_path, args.output, args.iou, args.lamda)
	print("Evaluation Finished \n **************************** ")

	

	
	

