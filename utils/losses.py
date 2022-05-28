import segmentation_models_pytorch as smp


def criterion(y_pred, y_true):
    bce = smp.losses.SoftBCEWithLogitsLoss()(y_pred, y_true)
    tve = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)(y_pred, y_true)
    return (bce + tve) / 2
