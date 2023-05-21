import torchvision
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from NST_project_settings import DEVICE, SAVED_IMAGE_DIR
import utils

DEVICE = utils.get_device()
def get_image_tensor(image_path, device=DEVICE, transform=None, add_batch_dim=False, batch_dim_index=0):
    image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).to(device)
    # [channels, height, width].

    image = image / 255.0

    if add_batch_dim:
        image = torch.unsqueeze(image, batch_dim_index)
        # [batch_size, channels, height, width]

    if transform is not None:
        image = transform(image)

    print(image_path, '- ', image.shape, device)

    return image


# Reference: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
def display_image(tensor_image, batch_dim_exist=False, batch_dim_index=0, save_image=False, file_name='saved_img.png'):
    if batch_dim_exist:
        plt.imshow(tensor_image.squeeze(dim=batch_dim_index).permute(1, 2,
                                                                     0))  # remove batch dim and Make the Channel dim last
    else:
        plt.imshow(tensor_image.permute(1, 2, 0))  # Make the Channel dim last

    if save_image:
        plt.savefig(SAVED_IMAGE_DIR + file_name, bbox_inches='tight')
    else:
        plt.show()


def get_gram_matrix(x):
    x = x.unsqueeze(0)
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    gram /= ch * h * w
    return gram


def calc_content_loss(content_image_feature_map, generated_image_feature_map):
    return torch.nn.MSELoss(reduction='mean')(content_image_feature_map, generated_image_feature_map)

def calc_style_loss(style_image_gram_matrix, generated_image_gram_matrix):
    return torch.nn.MSELoss(reduction='sum')(style_image_gram_matrix, generated_image_gram_matrix)


def calc_total_style_loss(style_feature_maps: list, generated_feature_maps: list):
    total_style_loss = 0

    feature_maps = zip(style_feature_maps, generated_feature_maps)
    for style_feature_map, generated_feature_map in feature_maps:
        style_image_gram_matrix = get_gram_matrix(style_feature_map)
        generated_image_gram_matrix = get_gram_matrix(generated_feature_map)

        total_style_loss += calc_style_loss(style_image_gram_matrix, generated_image_gram_matrix)

    return total_style_loss


def calc_variation_loss(generated_image):
    return torch.sum(torch.abs(generated_image[:, :, :, :-1] - generated_image[:, :, :, 1:])) + \
        torch.sum(torch.abs(generated_image[:, :, :-1, :] - generated_image[:, :, 1:, :]))
