from matplotlib import pyplot as plt
import torch

def plot_pred(model, ds, device):
    model.eval()
    with torch.inference_mode():
        ex_img, ex_seg, pat_id, slice_id = ds
        ex_img = ex_img.unsqueeze(0)
        pred = model(ex_img.to(device)).softmax(dim=1).argmax(dim=1).to(torch.uint8).cpu()
    title = {0: 'GroundTrouth', 1: 'Pred'}
    nrows, ncols = 1, 2  # array of sub-plots
    figsize = [14, 14]     # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        if i == 1:
            img = pred.permute(1,2,0) 
        else:
            img = ex_seg[i]
        axi.set_title(title[i])
        axi.imshow(img, cmap='gray')

    plt.show()
    print('Patient_ID: ', pat_id)
    print('Slice_ID: ', slice_id)

def plot_images(ds):
    ex_img, ex_seg, pat_id, slice_id = ds
    print(ex_img.shape)

    title = {0: 'Flair', 1: 'DWI', 2: 'T1', 3: 'T2', 4: 'Seg'}
    nrows, ncols = 1, 5  # array of sub-plots
    figsize = [14, 14]     # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        if i == 4:
            img = ex_seg[0]
        else:
            img = ex_img[i]
        axi.set_title(title[i])
        axi.imshow(img, cmap='gray')


    #plt.tight_layout(True)
    plt.show()
    print('Patient_ID: ', pat_id)
    print('Slice_ID: ', slice_id)

def plot_loss(epoch_loss, val_loss):
    # visualize the losses
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    ax.plot(epoch_loss, 'r-', linewidth=2, label='Train')     # training loss in red
    ax.plot(val_loss, 'b-', linewidth=2, label='Validation')  # vlaidation loss in blue
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend()
    fig.tight_layout()
    plt.show()