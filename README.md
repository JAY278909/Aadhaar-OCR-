# Aadhaar OCR Using AWS Reckognition

Extract data from aadhaar (front / back) using AWS Reckognition you can use pytessaract too but for better results i would suggest use recokgnition


### Requirenments 
 - AWS Reckognition Credentials would be required
 - Python Version >3.5 
 - GPU Enabled Machine(Not required but time to generate an output may varry depending on system)
 - PIP to be on latest version in order to install requirenment file 
 - You can use main module anywhere in your project
 - Please download model file (sorry but models are large in size so uploaded seperately on gdrive) and extract in root of directory

### Installation Guide
```bash
pip3 install -r /path/to/req2.txt
```
### Model File Path
```
https://drive.google.com/file/d/1c6nYJ-tX0aZjvQHLG_W5u5KW7hjiXMQB/view?usp=sharing
```
### Usage
```
Add keys aws reckognition keys in execute.py 
python3 exeute.py 
```



