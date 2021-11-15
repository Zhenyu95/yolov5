import coremltools as ct
import numpy as np
import PIL.Image
import time
import torch
import os

Height = 2560  # use the correct input image height
Width = 2560  # use the correct input image width
img_path = '/Users/zhenyu/Desktop/val/'

def load_image(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    img_np = np.array(img).astype(np.float32)
    return img_np, img

# _, img = load_image('/Users/zhenyu/Box/MLProject:IphoneAOI/datasets/data_4032*3024_20211105/images/train/0CELJZKE4L.jpg', resize_to=(Width, Height))

# # Load the image and resize using PIL utilities.
def load_model():
    model = ct.models.MLModel('/Users/zhenyu/Documents/Scripts/IphoneAOI/yolov5/best_1106.mlmodel')
    # torchmodel = torch.load_state_dict(torch.load('best_1106.pt', map_location=torch.device('cpu')))
    # torchmodel.eval()
    return model

def coreml_pred(model, img):
    start_time = time.time()
    out_dict = model.predict({'image': img})
    print("--- coreml %s seconds ---" % (time.time() - start_time))
    return out_dict

def torch_pred():
    start_time = time.time()
    out_dict = torchmodel(img)
    print("--- torch %s seconds ---" % (time.time() - start_time))

model = load_model()
# img_list = [f for f in os.listdir(img_path) if f.endswith('.jpg')]
img_list = ['/Users/zhenyu/Desktop/val/0CEY89HZON.jpg']
for image in img_list:
    _, img = load_image(image, resize_to=(Width, Height))
    out_dict = coreml_pred(model,img)
# torch_pred()

# # Scenario 2: load an image from a NumPy array.
# shape = (Height, Width, 3)  # height x width x RGB
# data = np.zeros(shape, dtype=np.uint8)
# # manipulate NumPy data
# pil_img = PIL.Image.fromarray(data)
# out_dict = model.predict({'image': pil_img})