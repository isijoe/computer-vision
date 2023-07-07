import torch

def print_train_time(start:float,
                     end:float, 
                     device:torch.device=None):
  total_time = end - start
  print(f"\nTrain time on {device}: {total_time:.3f} seconds")
  return total_time