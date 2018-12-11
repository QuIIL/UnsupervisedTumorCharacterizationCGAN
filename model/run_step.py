
import torch
import torch.nn.functional as F

def train_step(engine, batch, info):
    extra_opt = info['extra_train_opt']
    net_g, optimizer_g = info['g_info']
    net_d, optimizer_d = info['d_info']

    # batch contain only RGB
    msks_imgs, real_imgs = batch # NHWC, batch must be 1 sample

    msks_imgs = msks_imgs.permute(0, 3, 1, 2) # to NCHW
    real_imgs = real_imgs.permute(0, 3, 1, 2) # to NCHW

    msks_imgs = msks_imgs.to('cuda').float() / 255.0
    real_imgs = real_imgs.to('cuda').float() / 255.0

    # -----------------------------------------------------------
    # NOTE: if optimizer = optim.Optimizer(net.parameters())
    # net.zero_grad() is same as optimizer.zero_grad(). If multiple
    # optimizer per part of models, net.zero_grad() clear all

    # * Train G network
    net_g.zero_grad() # not rnn so not accumulate

    fake_imgs = net_g(msks_imgs)
    # gan loss
    fake_input = torch.cat((fake_imgs, msks_imgs), 1)
    fake_pred = net_d(fake_input)

    # making dynamic adversarial ground truth
    true = torch.ones_like(fake_pred, requires_grad=False)
    fake = torch.zeros_like(fake_pred, requires_grad=False)

    loss_gan = F.mse_loss(fake_pred, true, reduction='mean')

    # pixel wise loss - L1
    loss_pixel_l1 = F.l1_loss(fake_imgs, real_imgs)
    l1_lambda = 100 if 'lambda' not in extra_opt else extra_opt['lambda']
    loss_g = loss_gan + l1_lambda * loss_pixel_l1

    loss_g.backward()
    optimizer_g.step()

    # -----------------------------------------------------------
    # * Train D network every N step training of G 
    if engine.state.iteration % extra_opt['generator_period'] == 0:
        net_d.zero_grad() # not rnn so not accumulate

        # generate a batch of images
        fake_imgs = net_g(msks_imgs)
        # prevent gradient to flow to generator
        fake_imgs = fake_imgs.detach() 

        real_input = torch.cat((real_imgs, msks_imgs), 1)
        fake_input = torch.cat((fake_imgs, msks_imgs), 1)
        real_validity = net_d(real_input)
        fake_validity = net_d(fake_input)

        loss_d_real = F.mse_loss(real_validity, true, reduction='mean')
        loss_d_fake = F.mse_loss(fake_validity, fake, reduction='mean')
        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        loss_d.backward()
        optimizer_d.step()

    # -----------------------------------------------------------
    # * Aggregrating the output for tracking

    msks_imgs = torch.cat([msks_imgs]*3, dim=1) # if input C=1
    tracked_images = torch.cat([msks_imgs, fake_imgs, real_imgs], dim=3)
    tracked_images = tracked_images.detach()
    # tensorboardX doesnt need to permute back to NHWC
    tracked_images = tracked_images[:2] # return just 2 image
    tracked_images[tracked_images < 0] = 0
    tracked_images[tracked_images > 1] = 1.0    
    tracked_images = (tracked_images * 255.0).type(torch.uint8)
    tracked_images = tracked_images.cpu()

    if engine.state.iteration % extra_opt['generator_period'] == 0:
        return {
            'scalar' : {
                'loss_gan'   : loss_gan.item(),
                'loss_g'     : loss_g.item(),

                'loss_d' : loss_d.item(),
                'loss_d_fake': loss_d_fake.item(),
                'loss_d_real': loss_d_real.item(),
            },        
            'images'     : tracked_images.numpy(),
        }
    else:
        return {
            'scalar' : {
                'loss_d' : loss_d.item(),
                'loss_d_fake': loss_d_fake.item(),
                'loss_d_real': loss_d_real.item(),
            },        
            'images'     : tracked_images.numpy(),
        }
