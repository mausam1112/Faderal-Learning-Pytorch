from core.FL_arch import client_model_train, server_model_gradient_aggregation
from core.test_model import test


def train(epochs, num_clients, global_model, train_loaders, valid_loaders, optimizers):
    for e in range(epochs):
        if e >= 0:
            client_models = [global_model for _ in range(num_clients)]
        loss = 0
        for idx in range(num_clients):
            loss += client_model_train(client_models[idx], train_loaders[idx], optimizers[idx])
        
        global_model, _ = server_model_gradient_aggregation(global_model, client_models)
        # global_model, client_models = server_model_gradient_aggregation(global_model, client_models)
        valid_loss, valid_acc = test(global_model, valid_loaders)

        print(f"Epoch {e+1}/{epochs}: Train {loss=}, Valid loss={valid_loss}, Valid Accuracy={valid_acc}")
    return global_model