import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        #if verbose & (batch_idx % log_interval == 0):
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #epoch, batch_idx * len(data), len(dataloader.dataset),
                #100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    train_loss, train_accuracy1, train_accuracy5 = eval(model, loss, train_loader, device, verbose)
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, train_loss, train_accuracy1, train_accuracy5, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        print("Epoch", epoch)
        train_loss_trainmode = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        
        train_loss, train_accuracy1, train_accuracy5 = eval(model, loss, train_loader, device, verbose)
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        
        row = [train_loss_trainmode, train_loss, train_accuracy1, train_accuracy5, test_loss, accuracy1, accuracy5]
        
        scheduler.step()
        rows.append(row)
    columns = ['train_loss_trainmode', 'train_loss', 'train_top1', 'train_top5', 'test_loss', 'test_top1', 'test_top5']
    return pd.DataFrame(rows, columns=columns)

# adapted from train_eval_loop, adds savepoint paramater,
# an integer defining how many epochs from end we want to save the model
def train_eval_loop_midsave(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, rewind):
    train_loss, train_accuracy1, train_accuracy5 = eval(model, loss, train_loader, device, verbose)
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, train_loss, train_accuracy1, train_accuracy5, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        print("Epoch", epoch)
        train_loss_trainmode = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        
        train_loss, train_accuracy1, train_accuracy5 = eval(model, loss, train_loader, device, verbose)
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        
        row = [train_loss_trainmode, train_loss, train_accuracy1, train_accuracy5, test_loss, accuracy1, accuracy5]
        
        scheduler.step()
        rows.append(row)

        if epoch == (epochs - rewind - 1):
            torch.save(model.state_dict(), 'model_pretrain_midway.pt')
    columns = ['train_loss_trainmode', 'train_loss', 'train_top1', 'train_top5', 'test_loss', 'test_top1', 'test_top5']
    return pd.DataFrame(rows, columns=columns)



