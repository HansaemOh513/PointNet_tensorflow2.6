import os

def install(text):
	os.system(text)

install("conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0")
install("pip install tensorflow==2.6.0")
install("pip install protobuf==3.20")
install("pip install keras==2.6.0")
install("pip install scikit-learn==1.0")
install("pip install matplotlib==3.4.3")
install("pip install scipy==1.7.1")
install("pip install numpy==1.19.5")
