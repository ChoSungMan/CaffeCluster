import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

import os

from PIL import Image

def get_vector(image_name):
	
	img = Image.open(image_name)
	img.show()

	model = models.resnet18(pretrained=True)
	layer = model._modules.get('avgpool')

	model.eval()

	#scaler = transforms.Scale((224,224))
	scaler = transforms.Resize((224,224))

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	to_tensor = transforms.ToTensor()
	
	scaled_img = scaler(img)
	tensor_img = to_tensor(scaled_img)
	norm_img = normalize(tensor_img)
	

	t_img = Variable(norm_img.unsqueeze(0))
	
	print(t_img.shape)

	embeddings = torch.zeros(512)
	
	def copy_data(m, i, o):
		embeddings.copy_(o.data)

	h = layer.register_forward_hook(copy_data)

	model(t_img)

	h.remove()

	return embeddings


pic1 = str(input("Input first image name\n"))
pic2 = str(input("Input second image name\n"))

pic1 = os.path.expanduser(pic1)
pic2 = os.path.expanduser(pic2)

pic1_vec = get_vector(pic1)
pic2_vec = get_vector(pic2)

print(pic1_vec)
print(pic2_vec)

