import torch
from tqdm.auto import tqdm
import time
import os

def run_iteration(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: torch.device,
                  do_backprob=True):
    loss_iter = 0
    for x, y, pid, sid in dataloader:
        # Data to device
        x, y = x.to(device), y.squeeze(dim=1).long().to(device)

        # 1. Forward pass
        y_logits = model(x) #.squeeze()
        y_pred = y_logits.softmax(dim=1).argmax(dim=1)

        #print(f"y_logits dtype: {y_logits.dtype} | y_true dtype: {y.dtype}")
        #print(f"y_logits shape: {y_logits.shape} | y_true shape: {y.shape}")
        # 2. Calculate the loss
        loss = loss_fn(y_logits, y)

        if do_backprob:
            # 3. Optimizer zero grad
            optimizer.zero_grad()
            # 4. Loss backward
            loss.backward()
            # 5. Optimizer step
            optimizer.step()

        loss_iter += loss.item()

    return loss_iter/len(dataloader)

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device):
  
  is_better = True
  prev_loss = [float('inf'), float('inf')]

  epoch_loss = torch.zeros(epochs)
  val_loss = torch.zeros(epochs)
  save_path = "checkpoints"
  os.makedirs(save_path, exist_ok=True)
  # start timer for full training
  #t_start = time.time()
  # training loop
  for epoch in range(epochs):
      # start timer for epoch
      t_epoch = time.time()
      print('Epoch {} from {}'.format(epoch+1, epochs))

      model.train()
      epoch_loss[epoch] = run_iteration(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=device)

      # Validation
      model.eval()
      with torch.inference_mode():
          val_loss[epoch] = run_iteration(model=model,
                                          dataloader=test_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device,
                                          do_backprob=False)

          delta_epoch = time.time() - t_epoch

          # print the current epoch's training and validation mean loss
          print('[{}] Training loss: {:.4f}'.format(epoch+1, epoch_loss[epoch]))
          print('[{}] Validation Loss: {:.4f}\t Time: {:.2f}s'.format(epoch+1, val_loss[epoch], delta_epoch))

          # check if current epoch's losses are better then best saved
          is_better = epoch_loss[epoch] < prev_loss[0] and val_loss[epoch] <= prev_loss[1]
          if is_better:
              # update best training and validation losses
              prev_loss[0] = epoch_loss[epoch]
              prev_loss[1] = val_loss[epoch]
              # save best model
          if epoch > 15:
              torch.save(model.state_dict(), './checkpoints/isles01.pt')
              print("\033[91m {}\033[00m" .format("Saved best model"))
          if epoch > 5 and val_loss[epoch] > 5:
            break
  #t_end = time.time()
  #print('Finished Training in {:.2f} seconds'.format(t_end))
  #print_train_time(t_start, t_end, device)        