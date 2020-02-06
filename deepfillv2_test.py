from deepfillv2_train import DeepFillV2


if __name__ == '__main__':
    from util import arguments
    from pytorch_lightning import Trainer
    from util import constants
    from pytorch_lightning.callbacks import ModelCheckpoint
    import os
    from util.logger import NullLogger

    args = arguments.parse_arguments()

    vis_dir = os.path.join(constants.RUNS_FOLDER, args.dataset, args.experiment, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(constants.RUNS_FOLDER, args.dataset, args.experiment), save_best_only=False, verbose=False, period=args.save_epoch)

    model = DeepFillV2(args)

    trainer = Trainer(gpus=[0], early_stop_callback=None, nb_sanity_val_steps=3, logger=NullLogger(), checkpoint_callback=checkpoint_callback, max_nb_epochs=args.max_epoch, check_val_every_n_epoch=2)

    trainer.test(model)
