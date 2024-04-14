import torch
import torch.nn.functional as F

def test(global_model, test_loader):
    global_model.eval()
    test_loss, correct_pred = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            # images, labels = images.cuda(), labels.cuda() # uncomment if using GPU
            output = global_model(images)
            test_loss += F.nll_loss(output, labels, reduction='sum').item() # add up loss from each batch
            pred = output.argmax(dim=1, keepdim=True)
            correct_pred += pred.eq(labels.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    acc = correct_pred / len(test_loader)

    return test_loss, acc