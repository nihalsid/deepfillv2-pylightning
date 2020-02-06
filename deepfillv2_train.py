import pytorch_lightning as pl
import os
from util import constants
from util.misc import count_parameters
import torch
import numpy as np
from model import get_generator
from model.InpaintSADiscriminator import InpaintSADiscriminator
from dataset import InpaintDataset
from util.loss import ReconstructionLoss
from util.transforms import ToNumpyRGB256
from PIL import Image


torch.backends.cudnn.benchmark = True


class DeepFillV2(pl.LightningModule):

    def __init__(self, args):
        super(DeepFillV2, self).__init__()
        self.hparams = args
        self.net_G = get_generator(args)
        self.net_D = InpaintSADiscriminator(args.input_nc)
        print('#Params Generator: ', f'{count_parameters(self.net_G) / 1e6}M')
        print('#Params Discriminator: ', f'{count_parameters(self.net_D) / 1e6}M')
        self.recon_loss = ReconstructionLoss(args.l1_c_h, args.l1_c_nh, args.l1_r_h, args.l1_r_nh)
        self.refined_as_discriminator_input = args.refined_as_discriminator_input
        self.visualization_dataloader = self.setup_dataloader_for_visualizations()

    def configure_optimizers(self):
        lr = self.hparams.lr
        decay = self.hparams.weight_decay
        opt_g = torch.optim.Adam(self.net_G.parameters(), lr=lr, weight_decay=decay)
        opt_d = torch.optim.Adam(self.net_D.parameters(), lr=4 * lr, weight_decay=decay)
        return [opt_g, opt_d], []

    @pl.data_loader
    def train_dataloader(self):
        dataset = InpaintDataset.InpaintDataset(self.hparams.dataset, "train", self.hparams.image_size, self.hparams.bbox_shape, self.hparams.bbox_randomness, self.hparams.bbox_margin, self.hparams.bbox_max_num, self.hparams.overfit)
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = InpaintDataset.InpaintDataset(self.hparams.dataset, "val", self.hparams.image_size, self.hparams.bbox_shape, self.hparams.bbox_randomness, self.hparams.bbox_margin, self.hparams.bbox_max_num, False)
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    @pl.data_loader
    def test_dataloader(self):
        dataset = InpaintDataset.FixedInpaintDataset(self.hparams.dataset, self.hparams.vis_dataset, self.hparams.image_size, self.hparams.fixed_bbox_idx)
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def setup_dataloader_for_visualizations(self):
        dataset = InpaintDataset.InpaintDataset(self.hparams.dataset, self.hparams.vis_dataset, self.hparams.image_size, self.hparams.bbox_shape, self.hparams.bbox_randomness, self.hparams.bbox_margin, self.hparams.bbox_max_num, False)
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def training_step(self, batch, batch_idx, optimizer_idx):
        image = batch['image']
        mask = batch['mask']
        if optimizer_idx == 0:
            # generator training
            coarse_image, refined_image = self.net_G(image, mask)
            reconstruction_loss = self.recon_loss(image, coarse_image, refined_image, mask)
            completed_image = refined_image * mask + image * (1 - mask)
            if self.refined_as_discriminator_input:
                discriminator_input = refined_image
            else:
                discriminator_input = completed_image
            d_fake = self.net_D(torch.cat((discriminator_input, mask), dim=1))
            gen_loss = -self.hparams.gen_loss_alpha * torch.mean(d_fake)
            total_loss = gen_loss + reconstruction_loss
            self.logger.log_generator_losses(self.global_step, gen_loss, reconstruction_loss)
            self.logger.log_total_generator_loss(self.global_step, total_loss)
            return {
                'loss': total_loss,
                'progress_bar': {
                    'gen_loss': gen_loss,
                    'recon_loss': reconstruction_loss
                }
            }
        if optimizer_idx == 1:
            d_real = self.net_D(torch.cat((image, mask), dim=1))
            coarse_image, refined_image = self.net_G(image, mask)
            completed_image = refined_image * mask + image * (1 - mask)
            if self.refined_as_discriminator_input:
                discriminator_input = refined_image
            else:
                discriminator_input = completed_image
            d_fake = self.net_D(torch.cat((discriminator_input.detach(), mask), dim=1))
            real_loss = torch.mean(torch.nn.functional.relu(1. - d_real))
            fake_loss = torch.mean(torch.nn.functional.relu(1. + d_fake))
            disc_loss = self.hparams.disc_loss_alpha * (real_loss + fake_loss)
            self.logger.log_total_discriminator_loss(self.global_step, disc_loss)
            self.logger.log_discriminator_losses(self.global_step, real_loss, fake_loss)
            return {
                'loss': disc_loss,
                'progress_bar': {
                    'd_real_loss': real_loss,
                    'd_fake_loss': fake_loss
                }
            }

    def generate_images(self, image, mask):
        coarse_image, refined_image = self.net_G(image, mask)
        completed_image = (refined_image * mask + image * (1 - mask)).cpu().numpy()
        coarse_image = coarse_image.cpu().numpy()
        refined_image = refined_image.cpu().numpy()
        masked_image = image * (1 - mask) + mask
        masked_image = masked_image.cpu().numpy()
        return masked_image, coarse_image, refined_image, completed_image

    def test_step(self, batch, batch_idx):
        torgb = ToNumpyRGB256(-1, 1)
        with torch.no_grad():
            masked_image, coarse_image, refined_image, completed_image = self.generate_images(batch['image'], batch['mask'])
            for j in range(batch['image'].size(0)):
                visualization = np.hstack([torgb(masked_image[j]), torgb(coarse_image[j]), torgb(refined_image[j]), torgb(completed_image[j]), torgb(batch['image'][j].cpu().numpy())])
                Image.fromarray(visualization).save(os.path.join(constants.RUNS_FOLDER, self.hparams.dataset, self.hparams.experiment, "visualization", batch["name"][j]+".jpg"))
        return {'test_loss': torch.FloatTensor([-1.0,])}

    def test_end(self, outputs):
        return {'test_loss': torch.FloatTensor([-1.0,])}

    def on_epoch_end(self):
        images = []
        coarse = []
        refined = []
        masked = []
        completed = []
        torgb = ToNumpyRGB256(-1, 1)

        with torch.no_grad():
            for t, batch in enumerate(self.visualization_dataloader):
                batch['image'] = batch['image'].cuda()
                batch['mask'] = batch['mask'].cuda()
                masked_image, coarse_image, refined_image, completed_image = self.generate_images(batch['image'], batch['mask'])

                for j in range(batch['image'].size(0)):
                    images.append(torgb(batch['image'][j].cpu().numpy()))
                    coarse.append(torgb(coarse_image[j]))
                    refined.append(torgb(refined_image[j]))
                    masked.append(torgb(masked_image[j]))
                    completed.append(torgb(completed_image[j]))

            visualization = np.hstack([np.vstack(masked), np.vstack(coarse), np.vstack(refined), np.vstack(completed), np.vstack(images)])
            self.logger.log_image(self.global_step, visualization)


if __name__ == '__main__':
    from util import arguments
    from util.logger import NestedFolderTensorboardLogger
    from pytorch_lightning import Trainer
    from util import constants
    from pytorch_lightning.callbacks import ModelCheckpoint
    import os

    args = arguments.parse_arguments()

    logger = NestedFolderTensorboardLogger(save_dir=os.path.join(constants.RUNS_FOLDER, args.dataset), name=args.experiment)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(constants.RUNS_FOLDER, args.dataset, args.experiment), save_best_only=False, verbose=False, period=args.save_epoch)

    model = DeepFillV2(args)

    trainer = Trainer(gpus=[0], early_stop_callback=None, nb_sanity_val_steps=3, logger=logger, checkpoint_callback=checkpoint_callback, max_nb_epochs=args.max_epoch, check_val_every_n_epoch=2)

    trainer.fit(model)
