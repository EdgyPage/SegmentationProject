
import torch
import os
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, ConcatDataset, Dataset, random_split

class TrainVal(Dataset):
    def __init__(self, imageDir, labelDir, imageTransform = None, labelTransform = None):
        super().__init__()
        self.imageDir = imageDir
        self.labelDir = labelDir
        self.imageTransform = imageTransform
        self.labelTransform = labelTransform

        self.imageFileNames = sorted(os.listdir(imageDir), key= sortByCaseThenFrame)
        self.labelFileNames = sorted(os.listdir(labelDir), key= sortByCaseThenFrame)
        
    def __len__(self):
        return len(self.imageFileNames)
    
    def __getitem__(self, index):
        imageName = self.imageFileNames[index]
        labelName = self.labelFileNames[index]

        imagePath = os.path.join(self.imageDir, imageName)
        labelPath = os.path.join(self.labelDir, labelName)

        #converts images to greyscale. Labels out to be greyscale already but it doesn't hurt to convert again.
        image = torchvision.io.read_image(imagePath, mode = torchvision.io.image.ImageReadMode.RGB).float()
        label = torchvision.io.read_image(labelPath, mode = torchvision.io.image.ImageReadMode.GRAY).float()

        if self.imageTransform:
            image = self.imageTransform(image)
        
        if self.labelTransform:
            label = self.labelTransform(label)

        #squeezes label image to flattened tensor for comparison
        #label = label.flatten()

        #label = torch.threshold(label, 254, 1) 



        return image, label

#thanks Chat GPT
def sortByCaseThenFrame(filePath):
    parts = filePath.split('_')
    caseNumber = int(parts[3])
    frameNumber = int(parts[5].split('.')[0])
    return (caseNumber, frameNumber)


def imageDirsToLoaders(imageDir : str, labelDir: str, batchSize :int = 32, trainSplit : float = .8, imageTransforms : list = None, labelTransforms : list = None):
    dataset = TrainVal(imageDir= imageDir, labelDir= labelDir, imageTransform= imageTransforms, labelTransform=labelTransforms)

    trainSize = int(trainSplit * len(dataset))
    valSize = len(dataset) - trainSize

    trainSet, valSet = random_split(dataset, [trainSize, valSize])

    trainLoader = DataLoader(trainSet, batch_size= batchSize, shuffle=True, num_workers= os.cpu_count(), pin_memory=True)
    valLoader = DataLoader(valSet,batch_size= batchSize, shuffle=True, num_workers= os.cpu_count(), pin_memory= True)

    return trainLoader, valLoader

def imageDirsToTestLoader(imageDir : str, labelDir: str, testCases: int = 4, imageTransforms : list = None, labelTransforms : list = None):
    
    dataset = TrainVal(imageDir= imageDir, labelDir= labelDir, imageTransform= imageTransforms, labelTransform=labelTransforms)
    testLoader = DataLoader(dataset, batch_size= int(len(dataset)/testCases), shuffle= False)

    return testLoader



