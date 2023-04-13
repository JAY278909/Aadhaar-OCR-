import sys
import os 
from Aadhaar_Extract import *
from PIL import Image
import io

# Update AWS keys 
getClass = Aadhaar_Extract(aws_key='',aws_secret='')
im = Image.open("aadhar-card_5cea241ca12c8.png")
img_byte_arr = io.BytesIO()
im.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

# parse aadhaar front 
print(getClass.parseAadhaarFront(img_byte_arr)) 

#parse aadhaar back 
print(getClass.parseAadhaarBack(img_byte_arr))
