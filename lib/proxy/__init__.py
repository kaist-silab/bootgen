from lib.proxy.regression import DropoutRegressor, EnsembleRegressor

def get_proxy_model(tokenizer,num_token,max_len,category='dropout'):

    if category=='dropout':
        proxy = DropoutRegressor(tokenizer,num_token,max_len)
    else:
        proxy = EnsembleRegressor(tokenizer,num_token,max_len)
    return proxy