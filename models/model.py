from models import DeterministicModel, UNet, BayesUNet, ResNet18, BayesResNet18

def get_model(model_type: str, layer_type: str, dataset: str, init_w: str,
              net_architecture: str, feature_size: int, output_size: int=1,               
              kernel_size: int=0, pooling_size: int=0, nonlinearity: str='relu',
              stride: int=1, bias: bool=False, padding_size: int=0, input_size: int=0,
              prior_mu: float=None, prior_variance: float=None, posterior_mu: float=None, 
              posterior_rho: float=None, task: str=None, batch_norm: bool=True, 
              initialize_params: dict=None, sparse: dict=None, fit_variance_only: bool=False,
              test: bool=False):
    
    in_channels = 1
    if initialize_params is not None:
        masks=initialize_params['masks']
        init_params=initialize_params['params']
    else:
        masks, init_params = [], []
    
    if dataset.upper() == 'LIDC': 
        in_channels = 1
        input_size = 128

    elif dataset.upper() == 'ISIC':
        in_channels = 3


    try: 
        if model_type == "deterministic": 
            if net_architecture.upper() == "RESNET18": 
                net =  ResNet18(input_size=input_size, feature_size=feature_size, output_size=output_size, 
                                kernel_size=kernel_size, stride=stride, pooling_size=pooling_size, 
                                nonlinearity=nonlinearity, sparse=sparse, padding=padding_size, add_bias=bias)
            elif net_architecture.upper() == "UNET":
                net =  UNet(in_channels=in_channels, output_size=output_size, feature_size=feature_size, batch_norm=batch_norm)            

        elif "bayesian" in model_type:
            if net_architecture.upper() == "RESNET18":
                net = BayesResNet18(input_size=input_size, feature_size=feature_size,  
                                output_size=output_size, add_bias=bias, kernel_size=kernel_size,
                                stride=stride, padding=padding_size, pooling_size=pooling_size,
                                prior_mu=prior_mu, prior_variance=prior_variance, 
                                posterior_mu=posterior_mu, posterior_rho=posterior_rho, 
                                sparse=sparse, masks=masks, init_params=init_params,
                                nonlinearity=nonlinearity, fit_variance_only=fit_variance_only, 
                                test=test) 
            if net_architecture.upper() == "UNET":
                net = BayesUNet(in_channels=in_channels, feature_size=feature_size, output_size=output_size, 
                                prior_mu=prior_mu, prior_variance=prior_variance, batch_norm=batch_norm,
                                posterior_mu=posterior_mu, posterior_rho=posterior_rho, 
                                sparse=sparse, masks=masks, init_params=init_params,
                                fit_variance_only=fit_variance_only, test=test, add_bias=bias)  
        return net, input_size, feature_size, output_size
    
    except: 
        raise Exception(f"Network Type <<{model_type}>> is not implemented in get_model()")
        
