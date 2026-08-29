from typing import Optional
import logging
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from model.model import Model
from model.utils import logging_config

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_and_test(
        self,
        model: "LightningModule",
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule: Optional["LightningDataModule"] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        """fit and test the model"""
        self.fit(model=model, 
        train_dataloaders=train_dataloaders, 
        val_dataloaders=val_dataloaders, 
        datamodule=datamodule, ckpt_path=ckpt_path)
        self.test(model=model, ckpt_path="best", datamodule=datamodule)

    def test(
        self,
        model=None,
        datamodule=None,
        ckpt_path=None,
        verbose=True,
        dataloaders=None,
    ):
        return super().test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            verbose=verbose,
            dataloaders=dataloaders,
        )

    def _run(self, model, ckpt_path=None, **kwargs):
        """Override _run to handle checkpoint loading with weights_only"""
        if ckpt_path is not None:
            # 加载checkpoint，但只加载模型权重
            import torch
            import logging
            
            logging.info(f"\n{'='*80}")
            logging.info(f"Loading checkpoint from: {ckpt_path}")
            logging.info(f"{'='*80}")
            
            # 手动加载checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            # 只保留state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                
                # 过滤不匹配的键
                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                
                for key, value in state_dict.items():
                    if key in model_state_dict and value.shape == model_state_dict[key].shape:
                        filtered_state_dict[key] = value
                    else:
                        logging.info(f"⚠️  Skipped: {key}")
                
                # 加载过滤后的权重
                model.load_state_dict(filtered_state_dict, strict=False)
                
                logging.info(f"✅ Loaded {len(filtered_state_dict)} / {len(state_dict)} parameters")
                logging.info(f"{'='*80}\n")
            
            # 不传递ckpt_path给父类，因为我们已经手动加载了
            ckpt_path = None
        
        return super()._run(model, ckpt_path=ckpt_path, **kwargs)


class MyCLI(LightningCLI):
    def __init__(self, **kwargs):
        super().__init__(run=True, trainer_class=MyTrainer, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data.init_args.disease_names", "model.init_args.disease_names"
        )
        parser.link_arguments(
            "data.init_args.lesion_names", "model.init_args.lesion_names"
        )
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.add_argument(
            "--config_overwrite",
            default=False,
            action="store_true",
            help="whether to overwrite the config file",
        )
        return super().add_arguments_to_parser(parser)

    def instantiate_classes(self) -> None:
        try:
            config_overwrite = self.config["config_overwrite"]
        except KeyError:
            config_overwrite = self.config[self.subcommand]["config_overwrite"]  # type: ignore
        if config_overwrite:
            print("Overwriting config file")
            self.save_config_kwargs = {"overwrite": True}
        super().instantiate_classes()
        logging_config(self.trainer.log_dir, self.trainer.local_rank)

    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        subcommands = LightningCLI.subcommands()
        subcommands.update(
            {
                "fit_and_test": {
                    "model",
                    "train_dataloaders",
                    "val_dataloaders",
                    "datamodule",
                },
            }
        )
        return subcommands


def cli_main():
    _ = MyCLI()


if __name__ == "__main__":
    # tensor core, start!!!!!!!!!
    import torch
    torch.set_float32_matmul_precision('medium')
    print("Welcome to this experiment. CUDA Tensor core is ready(medium).")
    cli_main()
