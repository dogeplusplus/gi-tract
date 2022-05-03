import pytest
import torch

from models.unet import UNet


@pytest.mark.parametrize("filters", [[32] * i for i in range(1, 6)])
def test_unet(filters):
    channels = 1
    num_classes = 3

    model = UNet(filters, channels, num_classes, kernel_size=(3, 3))

    x = torch.ones((1, 1, 512, 512))
    y = model(x)

    assert y.shape == (1, 3, 512, 512)
