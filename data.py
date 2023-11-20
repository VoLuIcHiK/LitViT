import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
PATH_DATASETS = './ViT'
BATCH_SIZE = 64


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.dims = (1, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        #скачивание данных
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        #разделение на train/val для использования даталоэдером
        if stage == "fit" or stage is None:
            full_train_set = CIFAR10(self.data_dir, train=True, transform=self.transform)
            #train_size = int(0.8 * len(full_train_set))
            #val_size = len(full_train_set) - train_size
            self.trainset, self.valset = random_split(full_train_set, [40000, 10000])

        #тестовый датасет для использования даталоэдером
        if stage == "test" or stage is None:
            self.testset = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=BATCH_SIZE)