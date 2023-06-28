import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class EvalDataset(Dataset):

    def __init__(self, images):
        self.images = images
        # self.transforms = Compose([
        #     ToTensor(),
        # ])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx]), torch.tensor([0])


def get_eval_dataloader(images, batch_size, num_workers):
    return DataLoader(EvalDataset(images), batch_size, False,
        num_workers=num_workers)
