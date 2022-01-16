import os
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from PIL import Image
from XKNet import resnet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

#定义自己的数据集合
class FlameSet(data.Dataset):
    def __init__(self, root):
        # 所有图片的绝对路径
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transform
        self.names = ['test/' + x for x in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        data = self.transforms(pil_img)
        names = self.names
        return data, names

    def __len__(self):
        return len(self.imgs)

# def evalute(best_model, loader):
#     best_model.eval()
#     correct = 0
#     total = len(loader.dataset)
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         with torch.no_grad():
#             logits = best_model(x)
#             pred = logits.argmax(dim=1)
#         correct += torch.eq(pred, y).sum().float().item()
#     return correct / total

# def prediect(img_path):
#     # net = torch.load('model.pkl')
#     net = resnet18()
#     net.load_state_dict(torch.load(r'model\best_model_08-51.pth'))
#     net = net.to(device)
#     torch.no_grad()
#     img = Image.open(img_path)
#     img = transform(img).unsqueeze(0)
#     outputs = net(img.to(device))
#     _, predicted = torch.max(outputs, 1)
#     # print(predicted)
#     print('this picture maybe :', classes[predicted[0]])


data_dir = r'C:\Users\ustc\OneDrive - mail.ustc.edu.cn\深度学习\homework2\load_test'
data_test = FlameSet(os.path.join(data_dir, 'test'))

test_list = os.listdir(os.path.join(data_dir, 'test'))
test_list = ['test/' + x for x in test_list]


# loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=True, num_workers=0)

net = resnet18(drop_out_prob=0)
# net.load_state_dict(torch.load('./model/best_model_15-53.pkl'))
net = torchvision.models.resnet18(pretrained=True)
net = net.to(device)
net.eval()
i = 0
for x in data_test:
    name = test_list[i]
    x = x.to(device)
    x = x.view([1, 3, 64, 64])
    torch.no_grad()
    logits = net(x)
    pred = logits.argmax(dim=1).item()
    with open('./test.txt', 'a') as f:
        f.writelines([name, ' ', str(pred), '\n'])
    i += 1

# evalute(model, dataloaders['val'])
for x in data_test:
    print(x)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
