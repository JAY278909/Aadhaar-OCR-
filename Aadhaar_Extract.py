#################################################################################################
#### Kissht_Ocr for Data Extraction from Documents 			 	                          ###
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import cv2
import os
from tensorflow.keras import backend as k
import sys
from PIL import Image ,TiffTags,ExifTags
from io import BytesIO
from datetime import datetime,timedelta
import boto3
import re
import json

class Aadhaar_Extract():

	# Aamazon reckognition 
	# Use your own credentials to get this work
	aws_key = ''
	aws_secret = ''
	def __init__(self,aws_key,aws_secret):

		self.aws_key= aws_key
		self.aws_secret=aws_secret

	def detectTextFromImage(self,ObjectBytes):

		rekognition = boto3.client("rekognition", aws_access_key_id=self.aws_key,aws_secret_access_key=self.aws_secret,region_name='ap-south-1')
		response = rekognition.detect_text(Image={'Bytes':ObjectBytes})

		return response

	def createDate(self,date_string,input_format):

			create_date = datetime.strptime(str(date_string).strip(),input_format)
			get_month = '0'+str(create_date.month) if len(str(create_date.month)) == 1 else str(create_date.month)
			get_month = '0'+str(create_date.day) if len(str(create_date.day)) == 1 else str(create_date.day)
			create_date = str(create_date.year)+'-'+str(get_month)+'-'+str(get_month)
			return create_date

	def parseAadhaarFront(self,image_bytes):

			response_textract = self.detectTextFromImage(image_bytes)
			get_detected_text = []
			get_confidence_primary = []
			other_text = []
			other_text_Confidence = []
			for textparser in response_textract['TextDetections']:

				# Get most confident textx like name , aadhar number etc
				if textparser['Confidence'] > 95 and textparser['Type'] == 'LINE':

					get_detected_text.append(textparser['DetectedText']) # Push text
					get_confidence_primary.append(textparser['Confidence']) # Push corresponding confidence

				# Get all other texts to fetct date , gender etc
				if 	textparser['Confidence'] < 100 and textparser['Type'] == 'LINE':

					other_text.append(textparser['DetectedText']) # Push text
					other_text_Confidence.append(textparser['Confidence']) # Push corresponding confidence


			filter_texts = []
			filter_index_primary = []
			# Filter out unwanted text from primary text array
			for textindex in range(len(get_detected_text)):

					if not re.match(r"w Government of India|Government Of India|Government|India|of India",get_detected_text[textindex],re.MULTILINE | re.DOTALL | re.IGNORECASE):
							filter_texts.append(get_detected_text[textindex])
							filter_index_primary.append(get_confidence_primary[textindex])


			# Fetch name
			get_name = {'conf':filter_index_primary[0],'value':filter_texts[0]}

			# remove name index and its confidence index
			del[filter_texts[0]]
			del[filter_index_primary[0]]

			# fetch birth year , date and gender
			only_birth_year = {}
			only_gender = {}
			only_birth_date = {}
			full_text = ''
			for textindex in range(len(other_text)):

				# fetch birth year
				get_year = re.search(r"\b(19|20)\d{2}\b",other_text[textindex],re.MULTILINE | re.DOTALL)

				if get_year:

					only_birth_year = {'conf':other_text_Confidence[textindex],'value':get_year[0]}

				# fetch birth date
				get_birth_date = re.search(r"\d{1,2}\/\d{1,2}\/\d{2,4}",other_text[textindex],re.MULTILINE | re.DOTALL)

				if get_birth_date:

					only_birth_date = {'conf':other_text_Confidence[textindex],'value':self.createDate(get_birth_date[0],"%d/%m/%Y")}

				# fetch gender
				Get_Gender = re.search(r"Male|Female",other_text[textindex],re.MULTILINE | re.DOTALL | re.IGNORECASE)

				if Get_Gender:
					only_gender = {'conf':other_text_Confidence[textindex],'value':Get_Gender[0]}

				# Append everything to make full text
				full_text += other_text[textindex]+'\n'

 			# Get aadhaar number this way because sometime aadhaar number comes in splited order
			get_aadhar = re.search(r"^\d{4}[ \n]\d{4}[ \n]\d{4}",full_text,re.MULTILINE | re.DOTALL)
			get_aadhaar_number = {}
			if get_aadhar:

				filter_aadhaar_number = re.sub(r"[\n ]",'',get_aadhar[0],re.MULTILINE | re.DOTALL | re.IGNORECASE)
				get_aadhaar_number = {'conf':filter_index_primary[0],'value':filter_aadhaar_number}


			# make response output array
			response_aadhaar_ocr = {}
			response_aadhaar_ocr['name'] = get_name
			response_aadhaar_ocr['gender'] = only_gender
			response_aadhaar_ocr['year_of_birth'] = only_birth_year
			response_aadhaar_ocr['dob'] = only_birth_date
			response_aadhaar_ocr['aadhaar_number'] = get_aadhaar_number
			response_aadhaar_ocr['is_xeros'] = 'no'
			response = {}
			#response['aadhaar_front'] = response_aadhaar_ocr
			response['success'] = 'True'
			response['aadhaar_front_data'] = json.dumps(response_aadhaar_ocr)

			return response


	def aadharBackdata(self,img_bytes):

		Nparr = np.fromstring(img_bytes, np.uint8)
		Img_Np = cv2.imdecode(Nparr, cv2.IMREAD_COLOR)

		#loading YoloV3 weights and configuration file with the help of dnn module of OpenCV
		net = cv2.dnn.readNet(os.getcwd() +'/Ml_Models/aadhaar_back_detenction/yolov3-custom_last.weights',os.getcwd() +'/Ml_Models/aadhaar_back_detenction/yolov3-custom.cfg')

		with open(os.getcwd() +"/Ml_Models/aadhaar_back_detenction/obj.names","r") as f:
			classes = f.read().splitlines()

		layers_names = net.getLayerNames()#returns the indices of the output layers of the network.
		output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
		# img_bytes=cv2.resize(img_bytes,None,fx=0.4,fy=0.4)
		height, width, channel =Img_Np.shape

		#accepting image  model and output layers as parameters.
		blob = cv2.dnn.blobFromImage(Img_Np , 0.0039, (416, 416), (0, 0, 0), True, crop=False)
		net.setInput(blob)
		outs = net.forward(output_layers)
		for out in outs:
			for detection in out:
				#identify the index of class with highest confidence/score
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]
				if confidence > 0.5:
					centre_x = int(detection[0] * width)#coordinates of the centre of the object detected
					centre_y = int(detection[1] * height)
					w = int(detection[2] * width)# height and width of the bounding box,
					h = int(detection[3] * height)
					x = int(centre_x - w / 2)
					y = int(centre_y - h / 2)
					crop_img = Img_Np [y:y + h, x:x + w]

					if crop_img is not None:
						return crop_img
					else:
						return False

	def parseAadhaarBack(self,image_bytes):

		states_list = ("Andhra Pradesh","Arunachal Pradesh ","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Andaman and Nicobar Islands","Chandigarh","Dadra and Nagar Haveli","Daman and Diu","Lakshadweep","National Capital Territory of Delhi","Puducherry")

		opencv_cropped_image = self.aadharBackdata(image_bytes)
		pillow_image = Image.fromarray(opencv_cropped_image)
		image_buffer = BytesIO()
		pillow_image.save(image_buffer,format='JPEG')
		face_match_response_text = self.detectTextFromImage(image_buffer.getvalue())
		get_detected_text = []
		get_confidence_primary = []
		other_text = []
		other_text_confidence = []
		get_address_nodes = {}
		__final_data_object = {}
		__final_data_object_1 = {}
		for textparser in face_match_response_text['TextDetections']:

			# Get most confident textx like name ,pan name , pan card number etc
			if textparser['Confidence'] < 100 and textparser['Type'] == 'LINE':

				get_detected_text.append(textparser['DetectedText']) # Push text
				get_confidence_primary.append(textparser['Confidence']) # Push corresponding confidence

				# Get all other texts to fetct date , gender etc
				if 	textparser['Confidence'] < 100 and textparser['Type'] == 'LINE':

					other_text.append(textparser['DetectedText']) # Push text
					other_text_confidence.append(textparser['Confidence']) # Push corresponding confidence

			Pan_Card_Type = ''

		filter_texts = []
		filter_index_primary = []
		# Filter out unwanted text from primary text array

		generate_string = ''
		for textindex in range(len(get_detected_text)):

				split_text_comma = get_detected_text[textindex].split(',')
				for texts in range(len(split_text_comma)):

					if re.search(r"Address|:",split_text_comma[texts],re.MULTILINE | re.DOTALL | re.IGNORECASE) :
						split_text_comma[texts] = re.sub(r"Address|:","",split_text_comma[texts],re.MULTILINE | re.DOTALL | re.IGNORECASE)



					if re.search(r"\d{6}",split_text_comma[texts],re.MULTILINE | re.DOTALL | re.IGNORECASE):
							get_pincode = re.search(r"\d{6}",split_text_comma[texts],re.MULTILINE | re.DOTALL | re.IGNORECASE)
							if get_pincode:
								get_address_nodes['pincode'] = get_pincode.group()
								__final_data_object['pincode'] = {'value':get_pincode.group(),'conf':get_confidence_primary[textindex]}
								split_text_comma[texts] = re.sub(r"\d{6}","",split_text_comma[texts],re.MULTILINE | re.DOTALL | re.IGNORECASE)

				generate_string += ",".join(split_text_comma)+' '


		full_text_address = generate_string+' '+__final_data_object['pincode'].get('value')

		generate_string = generate_string.replace('.',',')

		if re.search(r"C\/O|S\/O|M\/O|D\/O",generate_string,re.MULTILINE | re.DOTALL | re.IGNORECASE):

				get_care_of = re.search(r"D\/O(.*?),",generate_string,re.MULTILINE | re.DOTALL | re.IGNORECASE)

				if get_care_of:
					__final_data_object['care_of'] = {'value':get_care_of.group(),'conf':get_confidence_primary[textindex]}
					generate_string = re.sub(r"D\/O(.*?),","",generate_string,re.MULTILINE | re.DOTALL | re.IGNORECASE)


		for states in states_list:
			if re.search(r""+states,generate_string,re.MULTILINE | re.DOTALL | re.IGNORECASE):
				get_state = re.search(r""+states,generate_string,re.MULTILINE | re.DOTALL | re.IGNORECASE)
				if get_state:
					get_address_nodes['state'] = get_state.group()
					generate_string = re.sub(r""+states,"",generate_string,re.MULTILINE | re.DOTALL | re.IGNORECASE)

		split_string_array = generate_string.split(",")

		generate_string = re.sub(r"-|:",'',generate_string,re.MULTILINE | re.DOTALL | re.IGNORECASE)
		remaning_address_nodes = [i.strip() for i in generate_string.split(',') if i.strip()]

		get_address_nodes['city'] = remaning_address_nodes[-1]
		del remaning_address_nodes[-1]

		if len(remaning_address_nodes) > 0:

			if re.search(r"\d{1,6}",remaning_address_nodes[0],re.MULTILINE | re.DOTALL | re.IGNORECASE):
					get_address_nodes['flat_house'] = remaning_address_nodes[0]
					del remaning_address_nodes[0]

		address_nodes_array = ['line1','line2','street','landmark','locality']
		for remaing_nodes in range(len(address_nodes_array)):

			if remaing_nodes+1 <= len(remaning_address_nodes):
				get_address_nodes[address_nodes_array[remaing_nodes]] = remaning_address_nodes[remaing_nodes]
			else:
				get_address_nodes[address_nodes_array[remaing_nodes]] = ''


		get_address_nodes['full_text_address'] = full_text_address
		__final_data_object['address']=	get_address_nodes
		__final_data_object_1['success'] = 'True'
		__final_data_object_1['extracted_data'] = json.dumps(__final_data_object)

		return __final_data_object_1

