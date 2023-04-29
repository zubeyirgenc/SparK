from models.convnext import convnext_small, convnext_base, convnext_nano
# from timm.models.convnext import convnext_small
import torch


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


if __name__ == '__main__':
    nano_pretrained_path = '/home/zubeyir/Desktop/work/weights/convnext_nano_1kpretrained.pth'
    model = convnext_nano()
    checkpoint = torch.load(nano_pretrained_path)['module']
    load_state_dict(model, checkpoint)
    print('Nano done')

    small_pretrained_path = '/home/zubeyir/Desktop/work/weights/convnext_small_spark.pth'
    model = convnext_small()
    checkpoint = torch.load(small_pretrained_path)
    # model.load_state_dict(torch.load(pretrained_path))
    load_state_dict(model, checkpoint)
    print('Small done')

    base_pretrained_path = '/home/zubeyir/Desktop/work/weights/convnextv2_base_1k_224_ema.pt'
    base_model = convnext_base()
    checkpoint = torch.load(base_pretrained_path)['model']
    load_state_dict(base_model, checkpoint)
    print('Base done')
