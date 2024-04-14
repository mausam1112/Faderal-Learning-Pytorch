import torch
import torch.nn.functional as F


def client_model_train(client_model, train_loader, optimizer):
    client_model.train()
    for images, labels in train_loader:
        # images, labels = images.cuda(), labels.cuda() # uncomment if using GPU
        optimizer.zero_grad()
        output = client_model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    # print(f"{loss.item()}")
    return loss.item()


def server_model_gradient_aggregation(global_model, client_models):
    global_model_state = global_model.state_dict()
    for k in global_model_state.keys():
        global_model_state[k] = torch.stack(
            [client_models[i].state_dict()[k] for i in range(len(client_models))], 0
        ).mean(0)
    global_model.load_state_dict(global_model_state)
    client_models_updated = [model.load_state_dict(global_model.state_dict()) for model in client_models]
    return global_model, client_models_updated