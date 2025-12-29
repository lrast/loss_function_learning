# supporting functions for class token randomization in different layers of the network
import torch

from tqdm import tqdm


def model_accuracy(model, dataset):
    """ Want: shuffle, larger batch sizes so that shuffling tokens gives signal
    """
    dl = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    outputs = []
    
    for batch in tqdm(iter(dl)):
        images, labels = batch
        preds = model.forward(images.to('mps')).logits.argmax(1).cpu()
    
        outputs.append(preds.cpu() == labels)

    return torch.concat(outputs)


def randomize_cls_token(model, layer_name, dataset,
                        debug=False, randomization_mode=''):
    """ Randomize the class tokens in the specified layer, compute accuracy
    """

    module_dict = {k: v for k, v in model.named_modules()}
    module = module_dict[layer_name]

    def randomization_hook(module, input, output):
        if debug:
            print('in hook')
        copied_output = output.clone()
        batch, tokens, dims = copied_output.shape
        
        if randomization_mode == 'shuffle':
            ind_perm = torch.randperm(batch)
            copied_output[:, 0, :] = copied_output[ind_perm, 0, :]
        else:
            # default to normal
            copied_output[:, 0, :] = torch.randn(batch, dims)

        return copied_output

    handle = module.register_forward_hook(randomization_hook)
    outputs = model_accuracy(model, dataset)

    handle.remove()

    return outputs.float().mean().item()


def list_all_hooks(model):
    for name, module in model.named_modules():
        # Check forward hooks
        if module._forward_hooks or module._forward_pre_hooks:
            print(f"Module: {name}")
            print(f"  Forward Hooks: {list(module._forward_hooks.values())}")
            print(f"  Pre-Forward Hooks: {list(module._forward_pre_hooks.values())}")
        
        # Check backward hooks
        if module._backward_hooks or module._backward_pre_hooks:
            print(f"Module: {name}")
            print(f"  Backward Hooks: {list(module._backward_hooks.values())}")


def layer_sweep(model, dataset):
    pass



