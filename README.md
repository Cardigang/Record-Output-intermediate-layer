# Record-Output-intermediate-layer
A model is being developed that requires an output middle layer to document existing methods
A total of three ways to read the middle tier are recorded:
1. Common, simple, but expensive memory and computation
          for i in range(len(model)):
            x = model[i](x)
            if i == 2:
                ReLu_out = x
          print('ReLu_out.shape：\n\t',ReLu_out.shape)
2.Hook method: record the layer input and output, check the result
                class TestForHook(nn.Module):
                    def __init__(self):
                        super().__init__()

                        self.linear_1 = nn.Linear(in_features=2, out_features=2)
                        self.linear_2 = nn.Linear(in_features=2, out_features=1)
                        self.relu = nn.ReLU()
                        self.relu6 = nn.ReLU6()
                        self.initialize()

                    def forward(self, x):
                        linear_1 = self.linear_1(x)
                        linear_2 = self.linear_2(linear_1)
                        relu = self.relu(linear_2)
                        relu_6 = self.relu6(relu)
                        layers_in = (x, linear_1, linear_2)
                        layers_out = (linear_1, linear_2, relu)
                        return relu_6, layers_in, layers_out

                features_in_hook = []
                features_out_hook = []

                def hook(module, fea_in, fea_out):
                    features_in_hook.append(fea_in)
                    features_out_hook.append(fea_out)
                    return None

                net = TestForHook()
3. The most recommended, optimized, Torchvision method:
        class IntermediateLayerGetter(nn.ModuleDict):
            """
            Module wrapper that returns intermediate layers from a model
            It has a strong assumption that the modules have been registered
            into the model in the same order as they are used.
            This means that one should **not** reuse the same nn.Module
            twice in the forward if you want this to work.
            Additionally, it is only able to query submodules that are directly
            assigned to the model. So if `model` is passed, `model.feature1` can
            be returned, but not `model.feature1.layer2`.
            Arguments:
                model (nn.Module): model on which we will extract the features
                return_layers (Dict[name, new_name]): a dict containing the names
                    of the modules for which the activations will be returned as
                    the key of the dict, and the value of the dict is the name
                    of the returned activation (which the user can specify).
            """

            def __init__(self, model, return_layers):
                if not set(return_layers).issubset([name for name, _ in model.named_children()]):
                    raise ValueError("return_layers are not present in model")

                orig_return_layers = return_layers
                return_layers = {k: v for k, v in return_layers.items()}
                layers = OrderedDict()
                for name, module in model.named_children():
                    layers[name] = module
                    if name in return_layers:
                        del return_layers[name]
                    if not return_layers:
                        break

                super(IntermediateLayerGetter, self).__init__(layers)
                self.return_layers = orig_return_layers

            def forward(self, x):
                out = OrderedDict()
                for name, module in self.named_children():
                    x = module(x)
                    if name in self.return_layers:
                        out_name = self.return_layers[name]
                        out[out_name] = x
                return out
