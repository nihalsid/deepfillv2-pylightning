from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only
from torch.utils.tensorboard import SummaryWriter
import os


class NestedFolderTensorboardLogger(LightningLoggerBase):

    def __init__(self, save_dir, name, **kwargs):
        super().__init__()
        self.save_dir = save_dir
        self._name = name
        self.experiment_root = None
        self.experiment_discriminator = None
        self.experiment_generator = None
        self.kwargs = kwargs
        self.setup_experiments()

    def setup_experiments(self):
        root_dir = os.path.join(self.save_dir, self.name)
        os.makedirs(root_dir, exist_ok=True)
        generator_dir = os.path.join(self.save_dir, self.name, "generator")
        discriminator_dir = os.path.join(self.save_dir, self.name, "discriminator")
        os.makedirs(generator_dir, exist_ok=True)
        os.makedirs(discriminator_dir, exist_ok=True)
        self.experiment_root = SummaryWriter(log_dir=root_dir, **self.kwargs)
        self.experiment_discriminator = SummaryWriter(log_dir=discriminator_dir, **self.kwargs)
        self.experiment_generator = SummaryWriter(log_dir=generator_dir, **self.kwargs)

    @rank_zero_only
    def log_hyperparams(self, params):
        params = vars(params)
        self.experiment_root.add_hparams(hparam_dict=dict(params), metric_dict={})

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        return

    @rank_zero_only
    def log_image(self, step, visualization):
        self.experiment_root.add_image('visualization', visualization, step, dataformats='HWC')

    @rank_zero_only
    def log_generator_losses(self, step, gan_loss, reconstruction_loss):
        self.experiment_generator.add_scalars('G/losses', {
            'gan_loss': gan_loss,
            'reconstruction_loss': reconstruction_loss
        }, step)

    @rank_zero_only
    def log_discriminator_losses(self, step, real_loss, fake_loss):
        self.experiment_discriminator.add_scalars('D/losses', {
            'real': real_loss,
            'fake': fake_loss
        }, step)

    @rank_zero_only
    def log_total_generator_loss(self, step, loss):
        self.experiment_generator.add_scalar('total', loss, step)

    @rank_zero_only
    def log_total_discriminator_loss(self, step, loss):
        self.experiment_discriminator.add_scalar('total', loss, step)

    @rank_zero_only
    def save(self):
        self.experiment_generator._get_file_writer().flush()
        self.experiment_discriminator._get_file_writer().flush()
        self.experiment_root._get_file_writer().flush()

    @rank_zero_only
    def finalize(self, status):
        self.save()

    @property
    def version(self):
        return self._name

    @property
    def name(self):
        return self._name


class NullLogger(LightningLoggerBase):

    def __init__(self):
        super().__init__()
        self._name = 'null'

    @rank_zero_only
    def log_hyperparams(self, params):
        return

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        return

    @rank_zero_only
    def log_image(self, step, visualization):
        self.experiment_root.add_image('visualization', visualization, step, dataformats='HWC')

    @rank_zero_only
    def save(self):
        return

    @rank_zero_only
    def finalize(self, status):
        self.save()

    @property
    def version(self):
        return self._name

    @property
    def name(self):
        return self._name
