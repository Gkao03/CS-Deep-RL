
import torch
import datetime

import data
import model
import run
import train_config

config = train_config.get_arguments()

device = torch.device('cuda' if not config.nogpu else 'cpu')

trainvalset = data.ImageDataset(config.train_data_dirs, repeat=100)
if config.validate:
    trainset, valset = trainvalset.split(0.8, 0.2)
else:
    trainset = trainvalset
    valset = None
testset = data.ImageDataset(config.test_data_dirs, repeat=1)

trainset.set_patch(config.patch_size)
trainset.set_mode('train')
if config.validate:
    valset.set_patch(config.patch_size)
    valset.set_mode('val')
testset.set_patch(None)
testset.set_mode('test')

# create dataloader
trainldr = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, drop_last=True)
if config.validate:
    valldr = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=False)
else:
    valldr = None
testldr = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

net = model.ResidualCNN(gray=True).move(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.1)

log = run.train(net, optimizer, config.n_epoch, trainldr, config.sigma_n, validation=valldr, scheduler=scheduler, verbose=config.verbose)

test_mse = run.validate(net, testldr, config.sigma_n)
print('\nTest Set')
print('MSE: {:.5f}'.format(test_mse))

now = datetime.datetime.now()
curr_time = now.strftime("%m%d%y%H%M%S")
model_name = config.name + '_' + curr_time + '.pth'
torch.save(net.state_dict(), model_name)
