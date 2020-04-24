from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

def get_mnist_loader(batch_size, train, part=0, taskid=0, **loader_kwargs):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
        transforms.Lambda(lambda x: x.view([28, 28]))])

    dataset = datasets.MNIST(root='./data', download=True, transform=transform, train=train)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, **loader_kwargs)
    loader.taskid = taskid
    loader.name = 'MNIST_{}'.format(taskid, part)
    loader.short_name = 'MNIST'

    return loader
