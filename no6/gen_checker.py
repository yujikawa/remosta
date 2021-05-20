import torch
import torch.nn.functional as F
from model import Net
from torchvision import transforms

model_file_name = 'gen.pt'

net = Net()

net.load_state_dict(torch.load(model_file_name))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(img):
    x = transform(img)
    x = torch.tensor(x).unsqueeze(0)
    net.eval()
    y = F.softmax(net(x), 1)[0]
    return y[0].item()