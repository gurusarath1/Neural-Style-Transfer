import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable
import NST_project_settings
from NST_project_settings import DEVICE, SAVED_IMAGE_DIR
import utils
import NST_utils
from NST_utils import get_image_tensor, display_image, calc_content_loss, calc_variation_loss, calc_total_style_loss
from NST_model import Vgg16_truncated
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print('NST Main function !')

    DEVICE = utils.get_device()

    t = transforms.Compose(
        [
            # It was never required to normalize the input image
            # transforms.Normalize(NST_project_settings.IMAGE_MEANS, NST_project_settings.IMAGE_STDS),
            transforms.Resize(NST_project_settings.INPUT_IMAGE_SIZE),
        ]
    )
    content_image = get_image_tensor(NST_project_settings.CONTENT_IMAGE_PATH,
                                     device=DEVICE,
                                     transform=t,
                                     add_batch_dim=True)

    style_image = get_image_tensor(NST_project_settings.STYLE_IMAGE_PATH,
                                   device=DEVICE,
                                   transform=t,
                                   add_batch_dim=True)

    # display_image(content_image.cpu(), batch_dim_exist=True, save_image=True, file_name='content_image.png')
    # display_image(style_image.cpu(), batch_dim_exist=True, save_image=True, file_name='style_image.png')

    model = Vgg16_truncated().to(DEVICE).eval()  # Keep the model in eval mode, since we are not training the model
    # We only need the feature maps from a trained model

    # Image to optimize / Generated image / Target image
    target_image = Variable(content_image.detach().clone(),
                            requires_grad=True)  # Detach is needed so that there is no autograd relationship

    content_image_feature_maps = model(content_image)
    style_image_feature_maps = model(style_image)

    # Select a set of feature maps to calculate the content loss
    # This is our code specific
    get_vgg_feature_maps_index = {'conv3_64': 0, 'conv3_128': 1, 'conv3_256': 2, 'conv3_512': 3}
    use_feature_map_index = get_vgg_feature_maps_index['conv3_64']

    # We are only optimizing the generated image and not the model weights
    optimizer = torch.optim.Adam((target_image,), lr=NST_project_settings.LEARNING_RATE)

    for epoch in (range(NST_project_settings.NUM_EPOCHS)):

        try:
            optimizer.zero_grad()

            # Get the feature maps of the generated image by running it through the model
            target_image_feature_maps = model(target_image)

            # Calculate the content loss from a specific feature maps from a particular layer
            content_loss = calc_content_loss(content_image_feature_maps[use_feature_map_index],
                                             target_image_feature_maps[use_feature_map_index])

            # Calculate total style loss from all the feature maps from all the layers
            style_loss = calc_total_style_loss(target_image_feature_maps, style_image_feature_maps)

            # Calculate variation loss from the generated image
            variation_loss = calc_variation_loss(target_image)

            # Total loss = weighted sum of all the losses
            total_loss = (NST_project_settings.CONTENT_LOSS_WEIGHT * content_loss) + \
                         (NST_project_settings.STYLE_LOSS_WEIGHT * style_loss) + \
                         (NST_project_settings.VARIATION_LOSS_WEIGHT * variation_loss)

            # Calculate gradients
            total_loss.backward()

            # Apply gradient descent once
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Eopch = {epoch}  Loss = {total_loss.item()}')
                save_image = target_image.detach().cpu()
                torchvision.utils.save_image(save_image, SAVED_IMAGE_DIR + 'op_image_' + str(epoch) + '.png')

        # If the user presses ctrl-c while optimization is running save the output
        except KeyboardInterrupt:
            save_image = target_image.detach().cpu()
            torchvision.utils.save_image(save_image,
                                         SAVED_IMAGE_DIR + 'op_image_' + str(epoch) + '.png')

    save_image = target_image.detach().cpu()
    torchvision.utils.save_image(save_image, SAVED_IMAGE_DIR + '0_op_image_final.png')
