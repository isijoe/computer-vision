import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    # Put model in train mode
    model.train()

    # Setup the train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through DataLoader batches
    for batch, (img, label) in enumerate(dataloader):
        # Send data to target device
        img, label = img.to(device), label.to(device)

        # 1. Forward pass, make predictions (logits)
        pred_logits = model(img)

        # 2. Calculate and accumulate loss across all batches
        loss = loss_fn(pred_logits, label)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumalate accuracy across all batches
        pred = pred_logits.softmax(dim=1).argmax(dim=1)
        train_acc += pred.eq(label).sum().item()/len(label)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    # Put model in eval mode
    model.eval()
    # Setup the test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (img, label) in enumerate(dataloader):
            # Send data to target device
            img, label = img.to(device), label.to(device)

            # 1. Forward pass, make predictions (logits)
            pred_logits = model(img)

            # 2. Calculate and accumulate loss across all batches
            test_loss += loss_fn(pred_logits, label).item()

            # Calculate and accumulate accuracy across all batches
            pred = pred_logits.softmax(dim=1).argmax(dim=1)
            test_acc += pred.eq(label).sum().item()/len(label)

        # Adjust metrics to get average loss and accuracy per batch
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device):
    # Send model to target device
    model.to(device)

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch+1}\n------")
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.3f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.3f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results