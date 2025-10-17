

def insert_adapter_hook(layer, adapter):
    def hook_fn(module, input, output):
        return adapter(output)
    return layer.register_forward_hook(hook_fn)


target_module_name = "q_proj"
adapter_hooks = []

for name, module in model.named_modules():
    if name.endswith(target_module_name) and isinstance(module, nn.Linear):
        print(f"Inserting adapter at {name}")
        adapter = LoRAAdapter(hidden_dim=module.out_features)
        hook = insert_adapter_hook(module, adapter)
        adapter_hooks.append(hook)