import os
import torch
import scipy.io as sio
import glob
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# SPECIFY PATH TO THE DATASET
path_to_dataset = '../tiny-UCF101/'


def main():
    feature = []
    label = []
    categories = sorted(os.listdir(path_to_dataset))

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # FILL IN TO LOAD THE ResNet50 MODEL
    extractor = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    extractor.eval()

    for i, c in tqdm(enumerate(categories)):
        print(i,c)
        path_to_images = sorted(glob.glob(os.path.join(path_to_dataset, c) + '/*.jpg'))
        for p in path_to_images:

            # FILL IN TO LOAD IMAGE, PREPROCESS, EXTRACT FEATURES.
            # OUTPUT VARIABLE F EXPECTED TO BE THE FEATURE OF THE IMAGE OF DIMENSION (2048,)

            img = transform(Image.open(p))
            with torch.no_grad():
                F = extractor(img.unsqueeze(0))
            feature.append(F.detach().numpy()[0])
            label.append(categories.index(c))
    sio.savemat('ucf101dataset.mat', mdict={'feature': feature, 'label': label})


if __name__ == "__main__":
    main()
