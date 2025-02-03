import numpy as np
import torch
from . import utils
from collections import defaultdict

params = None


def init_globals(config):
    global params
    params = config
    params.device = params.model_params.device

    #set random seeds for reproducibility
    np.random.seed(params.model_params.seed)
    torch.manual_seed(params.model_params.seed)
    torch.cuda.manual_seed(params.model_params.seed)
    torch.backends.cudnn.deterministic = True

    from .elements.losses import StructureSimilarityIndexMap, get_reconstruction_loss, get_kl_loss
    from .elements.schedules import get_beta_schedule
    
    #params.free_loss = get_free_loss() if params.loss_params.custom_loss else None
    params.reconstruction_loss = get_reconstruction_loss()
    params.kl_divergence =  get_kl_loss()
    params.kldiv_schedule = get_beta_schedule()
    #params.gamma_schedule = get_gamma_schedule()
    params.ssim_metric =    StructureSimilarityIndexMap(image_channels=params.data_params.shape[0])
    return params


def training(config):
    p = init_globals(config)

    checkpoint, checkpoint_path = utils.load_experiment_for('train', p.log_params)
    logger = utils.setup_logger(checkpoint_path)
    device = p.model_params.device

    if checkpoint is not None:
        epoch = checkpoint.epoch + 1
        model = checkpoint.get_model()
        logger.info(f'Loaded Model Checkpoint from {p.log_params.load_from_train}')
    else:
        epoch = 0
        model = p.model_params.model()

    model.summary()
    model_parameters = filter(lambda param: param.requires_grad, model.parameters())
    logger.info(f'Number of trainable params '
                f'{np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000:.3f}m.')
    model = model.to(device)

    from .elements.optimizers import get_optimizer
    from .elements.schedules import get_schedule
    optimizer = get_optimizer(model=model,
                              type=p.optimizer_params.type,
                              learning_rate=p.optimizer_params.learning_rate,
                              beta_1=p.optimizer_params.beta1,
                              beta_2=p.optimizer_params.beta2,
                              epsilon=p.optimizer_params.epsilon,
                              weight_decay_rate=p.optimizer_params.l2_weight,
                              checkpoint=checkpoint)
    schedule = get_schedule(optimizer=optimizer,
                            decay_scheme=p.optimizer_params.learning_rate_scheme,
                            warmup_steps=p.optimizer_params.warmup_steps,
                            decay_steps=p.optimizer_params.decay_steps,
                            decay_rate=p.optimizer_params.decay_rate,
                            decay_start=p.optimizer_params.decay_start,
                            min_lr=p.optimizer_params.min_learning_rate,
                            last_epoch=torch.tensor(epoch-1),
                            checkpoint=checkpoint)

    dataset = p.data_params.dataset(**p.data_params.params)
    train_loader = dataset.get_train_loader(p.train_params.batch_size)
    val_loader = dataset.get_val_loader(p.eval_params.batch_size)
    
    # for jepa architectures, bit ugly solution 
    if p.loss_params.custom_loss is not None and isinstance(p.loss_params.custom_loss, list):
        from .elements.schedules import EMAMomentumSchedule
        p.ema_momentum_schedule = EMAMomentumSchedule(ipe=len(train_loader), num_epochs=p.train_params.total_train_epochs,
                                                      ema0=0.996, ema1=1.0)

    if p.train_params.unfreeze_first:
        model.unfreeeze()
    if len(p.train_params.freeze_nets) > 0:
        model.freeze(p.train_params.freeze_nets)

    from .functional import train
    train(model, optimizer, schedule, train_loader, val_loader, epoch, checkpoint_path, logger)
    return model


def testing(config):
    p = init_globals(config)

    checkpoint, checkpoint_path = utils.load_experiment_for('test', p.log_params)
    device = p.model_params.device

    assert checkpoint is not None
    model = checkpoint.get_model()
    print(f'Model Checkpoint is loaded from {p.log_params.load_from_eval}')

    model.summary()
    model = model.to(device)

    dataset = p.data_params.dataset(**p.data_params.params)
    test_loader = dataset.get_test_loader(p.eval_params.batch_size)

    from .functional import reconstruct
    results = reconstruct(
        net=model,
        dataset=test_loader,
        use_mean=p.eval_params.use_mean,
        offline_test=True
    )
    return results


class Hyperparams:
    def __init__(self, **config):
        self.config = defaultdict(**config)

    def __getattr__(self, name):
        if name == 'config':
            return super().__getattribute__(name)
        return self.config[name]

    def __setattr__(self, name, value):
        if name == 'config':
            super().__setattr__(name, value)
        else:
            self.config[name] = value

    def __getstate__(self):
        return self.config

    def __setstate__(self, state):
        self.config = state

    def keys(self):
        return self.config.keys()

    def __getitem__(self, item):
        return self.config[item]

    def to_json(self):
        from types import FunctionType
        from .elements.dataset import DataSet

        def convert_to_json_serializable(obj):
            if isinstance(obj, Hyperparams):
                return convert_to_json_serializable(obj.config)
            if isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            if isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            if callable(obj) or isinstance(obj, FunctionType):
                return str(obj)
            if isinstance(obj, DataSet):
                return str(obj)
            return obj

        json_serializable_config = convert_to_json_serializable(self.config)
        return json_serializable_config

    @classmethod
    def from_json(cls, json_str):
        import json
        data = json.loads(json_str)
        return cls(**data)

    @staticmethod
    def from_dict(dictionary):
        return Hyperparams(**{k: Hyperparams.from_dict(v) if isinstance(v, dict) else v
                              for k, v in dictionary.items()})


def load_model(path, loaded_params=None):
    checkpoint = utils.load_model(path)
    if loaded_params is not None:
        init_globals(loaded_params)
    return checkpoint.model
