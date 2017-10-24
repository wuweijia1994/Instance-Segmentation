import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode
#
# root_dir_wuweijia = './20171020_morning/'

def show_image(image, groundtruth_image, ix):
    ax = plt.subplot(1, 4, ix + 1)
    plt.tight_layout()
    ax.set_title('image #{}'.format(ix))
    ax.axis('off')
    plt.imshow(image)

    plt.pause(0.001)  # pause a bit so that plots are updated

class number_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 27 * 27
        # return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'image_' + str(idx) + '.jpg')
        image = io.imread(img_name)
        groundtruth_image = np.zeros([54, 54], np.uint8).reshape([54 * 54])
        for i in range(4):
            temp_image_name = os.path.join(self.root_dir, 'image_' + str(idx) +'_gt_' + str(i) + '.npy')
            new_image = np.load(temp_image_name)
            new_image = new_image[:, :, 0].reshape([54 * 54])
            groundtruth_image = groundtruth_image + new_image

        sample = {'image': image, 'groundtruth_image': groundtruth_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

# face_dataset = FaceLandmarksDataset(root_dir=root_dir_wuweijia)
#
# fig = plt.figure()
#
# for i in range(len(face_dataset)):
#     sample = face_dataset[i]
#
#     print(i, sample['image'].shape, sample['groundtruth_image'].shape)
#     sample.update({'ix':i})
#     show_image(**sample)
#
#     if i == 3:
#         plt.show()
#         break

class Slice(object):
    def __init__(self, output_layers = 0):
        assert isinstance(output_layers, (int, tuple))
        self.output_layers = output_layers

    def __call__(self, sample):
        image, groundtruth_image = sample['image'], sample['groundtruth_image']
        image = image[:, :, self.output_layers]
        return {'image': image, 'groundtruth_image': groundtruth_image}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, groundtruth_image = sample['image'], sample['groundtruth_image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'groundtruth_image': torch.from_numpy(groundtruth_image)}

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        image = tensor['image']
        # TODO: make efficient
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor


# transformed_dataset = FaceLandmarksDataset(root_dir=root_dir_wuweijia, transform=transforms.Compose([ToTensor()]))
#
# for i in range(len(transformed_dataset)):
#     sample = transformed_dataset[i]
#
#     print(i, sample['image'].size(), sample['groundtruth_image'].size())
#
#     if i == 3:
#         break
#
#
# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=4)


# Helper function to show a batch
def show_image_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, groundtruth_image = \
            sample_batched['image'], sample_batched['groundtruth_image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),
#           sample_batched['groundtruth_image'].size())
#
#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_image_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break



# import torch
# from torchvision import transforms, datasets
#
# data_transform = transforms.Compose([
#         # transforms.RandomSizedCrop(224),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
# hymenoptera_dataset = datasets.ImageFolder(root=root_dir_wuweijia,
#                                            transform=data_transform)
# dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                              batch_size=4, shuffle=True,
#                                              num_workers=4)

