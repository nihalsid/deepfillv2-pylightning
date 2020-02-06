import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='matterport', help='dataset name')
    parser.add_argument('--num_workers', type=int, default=2, help='num workers')
    parser.add_argument('--image_size', type=int, default=256, help='input image size')
    parser.add_argument('--bbox_shape', type=int, default=48, help='random box size')
    parser.add_argument('--bbox_randomness', type=float, default=0.25, help='variation in box size')
    parser.add_argument('--bbox_margin', type=int, default=32, help='margin from boundaries for box')
    parser.add_argument('--bbox_max_num', type=int, default=2, help='max num of boxes')
    parser.add_argument('--fixed_bbox_idx', type=int, default=0, help='boxes to be used for testing')
    parser.add_argument('--vis_dataset', type=str, default='vis_0', help='images to be visualized after each epoch')
    parser.add_argument('--overfit', default=False, action='store_true')

    # train params
    parser.add_argument('--max_epoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--save_epoch', type=int, default=5, help='save every nth epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.001')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay.')
    parser.add_argument('--l1_c_h', type=float, default=1.2, help='reconstruction coarse weight for holes')
    parser.add_argument('--l1_c_nh', type=float, default=1.2, help='reconstruction coarse weight for non-holes')
    parser.add_argument('--l1_r_h', type=float, default=1.2, help='reconstruction coarse weight for holes')
    parser.add_argument('--l1_r_nh', type=float, default=1.2, help='reconstruction coarse weight for non-holes')
    parser.add_argument('--gen_loss_alpha', type=float, default=0.005, help='generator loss lambda')
    parser.add_argument('--disc_loss_alpha', type=float, default=1., help='discriminator loss lambda')
    parser.add_argument('--refined_as_discriminator_input', default=False, action='store_true')
    parser.add_argument('--gen_type', type=str, choices=['GSAGAN', 'UNet'], default='GSAGAN')
    parser.add_argument('--unet_num_downs', type=int, default=8)
    parser.add_argument('--no_leaky_relu', default=False, action='store_true')
    parser.add_argument('--no_gated_conv', default=False, action='store_true')
    parser.add_argument('--no_attention', default=False, action='store_true')
    parser.add_argument('--no_intermediate_input_filling', default=False, action='store_true')

    # other
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--input_nc', type=int, default=4, help='number of input channels + mask')
    parser.add_argument('--experiment', type=str, default='fast_dev', help='experiment directory')
    parser.add_argument('--visualization_set', type=str, default='vis_0', help='validation samples to be visualized')

    return parser.parse_args()
