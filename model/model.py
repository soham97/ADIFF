def get_model_class(model_type):
    if model_type == 'ADiff':
        from model.adiff import ADiff
        return ADiff
    else:
        raise NotImplementedError
