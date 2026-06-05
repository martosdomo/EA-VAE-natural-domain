from hvae_backbone import Hyperparams

def get_hparams():
    # SET WHICH params TO USE HERE
    # |    |    |    |    |    |
    # v    v    v    v    v    v
    # import models.StandardVAE_b_2 as params
    import models.EAVAE_lognormal_b1_0_5_b2_0_5 as params

    config = Hyperparams(
        log_params=params.log_params,
        model_params=params.model_params,
        data_params=params.data_params,
        train_params=params.train_params,
        optimizer_params=params.optimizer_params,
        loss_params=params.loss_params,
        eval_params=params.eval_params,
        analysis_params=params.analysis_params,
    )

    return config