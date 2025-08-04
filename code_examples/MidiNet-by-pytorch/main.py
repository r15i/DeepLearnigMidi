import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import ipdb
import torchvision.utils as vutils
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import *
from ops import *


class get_dataloader(object):
    def __init__(self, data, prev_data, y):
        self.size = data.shape[0]
        self.data = torch.from_numpy(data).float()
        self.prev_data = torch.from_numpy(prev_data).float()
        self.y = torch.from_numpy(y).float()

        # self.label = np.array(label)

    def __getitem__(self, index):
        return self.data[index], self.prev_data[index], self.y[index]

    def __len__(self):
        return self.size


def load_data():
    #######load the data########
    check_range_st = 0
    check_range_ed = 129
    pitch_range = check_range_ed - check_range_st - 1
    # print('pitch range: {}'.format(pitch_range))

    X_tr = np.load("your training x")
    prev_X_tr = np.load("your training prev x")
    y_tr = np.load("your training chord")
    X_tr = X_tr[:, :, :, check_range_st:check_range_ed]
    prev_X_tr = prev_X_tr[:, :, :, check_range_st:check_range_ed]

    # test data shape(5048, 1, 16, 128)
    # train data shape(45448, 1, 16, 128)

    train_iter = get_dataloader(X_tr, prev_X_tr, y_tr)
    kwargs = {"num_workers": 4, "pin_memory": True}  # if args.cuda else {}
    train_loader = DataLoader(train_iter, batch_size=72, shuffle=True, **kwargs)

    print("data preparation is completed")
    #######################################
    return train_loader


def main():
    is_train = 1
    is_draw = 0
    is_sample = 0

    epochs = 20
    lr = 0.0002

    check_range_st = 0
    check_range_ed = 129
    pitch_range = check_range_ed - check_range_st - 1

    device = torch.device("cuda")
    train_loader = load_data()

    if is_train == 1:
        # training loop
        netG = generator(pitch_range).to(device)
        netD = discriminator(pitch_range).to(device)
        # set the networks in training mode
        netD.train()
        netG.train()
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

        batch_size = 72
        nz = 100
        # A fixed noise vector is created. This is used later to generate and save the same set of fake
        # images at regular intervals, allowing for a consistent visualization of the generator's progress.
        fixed_noise = torch.randn(batch_size, nz, device=device)
        # Standard labels for the GAN training loop.
        # The discriminator is trained to output 1 for real data and 0 for fake data.
        real_label = 1
        fake_label = 0
        average_lossD = 0
        average_lossG = 0
        average_D_x = 0
        average_D_G_z = 0

        lossD_list = []
        lossD_list_all = []
        lossG_list = []
        lossG_list_all = []
        D_x_list = []
        D_G_z_list = []

        # training loop
        for epoch in range(epochs):
            sum_lossD = 0
            sum_lossG = 0
            sum_D_x = 0
            sum_D_G_z = 0
            # This inner loop iterates through the data in batches.
            # Each iteration loads a batch of data (the real music bar X),
            # prev_data (the previous bar's melody), and chord (the chord condition y).
            for i, (data, prev_data, chord) in enumerate(train_loader, 0):
                ############################
                # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real

                # clears the previosly computed gradients
                netD.zero_grad()
                # data are moved to the gpu
                real_cpu = data.to(device)
                prev_data_cpu = prev_data.to(device)
                chord_cpu = chord.to(device)

                batch_size = real_cpu.size(0)

                # Creates a tensor of labels, all set to real_label
                label = torch.full((batch_size,), real_label, device=device)

                # forward pass on discriminator
                # The real data and chord condition are passed to the discriminator.
                # It returns the final sigmoid output (D),
                # the raw logits before the sigmoid (D_logits),
                # and a feature map (fm) from an intermediate layer.
                # This feature map will be used for feature matching later.
                D, D_logits, fm = netD(real_cpu, chord_cpu, batch_size, pitch_range)

                #####loss
                # Calculates the loss for the real dat
                # Calculates the loss for the real data.
                # sigmoid_cross_entropy_with_logits is a robust loss function for this task.
                # The labels are set to 0.9*torch.ones_like(D) for one-sided label smoothing,
                # which helps stabilize GAN training
                d_loss_real = reduce_mean(
                    sigmoid_cross_entropy_with_logits(
                        D_logits, 0.9 * torch.ones_like(D)
                    )
                )
                # Computes the gradients for this part of the loss.

                # retain_graph=True is used because the graph will be needed again for the fake data loss.
                d_loss_real.backward(retain_graph=True)
                D_x = D.mean().item()
                sum_D_x += D_x

                # train with fake

                # new noise
                noise = torch.randn(batch_size, nz, device=device)
                # use generator to create example
                fake = netG(noise, prev_data_cpu, chord_cpu, batch_size, pitch_range)

                label.fill_(fake_label)

                # passed by the discriminator
                D_, D_logits_, fm_ = netD(
                    fake.detach(), chord_cpu, batch_size, pitch_range
                )
                # Calculates the loss for the fake data, with labels set to 0.
                d_loss_fake = reduce_mean(
                    sigmoid_cross_entropy_with_logits(D_logits_, torch.zeros_like(D_))
                )
                # compute gradients for the fake
                d_loss_fake.backward(retain_graph=True)
                D_G_z1 = D_.mean().item()
                errD = d_loss_real + d_loss_fake
                errD = errD.item()
                lossD_list_all.append(errD)
                sum_lossD += errD
                # Updates the weights of the discriminator using the combined gradients from both the real and fake data.
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                # Clears the gradients for the generator.
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # passw the fake data throug the discriminator
                D_, D_logits_, fm_ = netD(fake, chord_cpu, batch_size, pitch_range)

                ###loss
                g_loss0 = reduce_mean(
                    sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_))
                )
                # Feature Matching
                # loss

                # regularization term that encourages the generator to produce data with similar statistical
                # properties to the real data, specifically in the feature space of the discriminator's layers.

                # Calculates the L2 loss between the mean feature maps of the real and fake data.
                features_from_g = reduce_mean_0(fm_)
                features_from_i = reduce_mean_0(fm)
                fm_g_loss1 = torch.mul(l2_loss(features_from_g, features_from_i), 0.1)

                # Calculates the L2 loss between the mean of the real and fake images themselves.
                mean_image_from_g = reduce_mean_0(fake)
                smean_image_from_i = reduce_mean_0(real_cpu)
                fm_g_loss2 = torch.mul(
                    l2_loss(mean_image_from_g, smean_image_from_i), 0.01
                )

                # The total generator loss is the sum of the adversarial loss and the two feature matching losses.
                errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                errG.backward(retain_graph=True)
                D_G_z2 = D_.mean().item()
                optimizerG.step()

                ############################
                # (3) Update G network again: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                D_, D_logits_, fm_ = netD(fake, chord_cpu, batch_size, pitch_range)

                ###loss
                g_loss0 = reduce_mean(
                    sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_))
                )
                # Feature Matching
                features_from_g = reduce_mean_0(fm_)
                features_from_i = reduce_mean_0(fm)
                loss_ = nn.MSELoss(reduction="sum")
                feature_l2_loss = loss_(features_from_g, features_from_i) / 2
                fm_g_loss1 = torch.mul(feature_l2_loss, 0.1)

                mean_image_from_g = reduce_mean_0(fake)
                smean_image_from_i = reduce_mean_0(real_cpu)
                mean_l2_loss = loss_(mean_image_from_g, smean_image_from_i) / 2
                fm_g_loss2 = torch.mul(mean_l2_loss, 0.01)
                errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                sum_lossG += errG
                errG.backward()
                lossG_list_all.append(errG.item())

                D_G_z2 = D_.mean().item()
                sum_D_G_z += D_G_z2
                optimizerG.step()

                # evry 5 epoch
                if epoch % 5 == 0:
                    print(
                        "[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                        % (
                            epoch,
                            epochs,
                            i,
                            len(train_loader),
                            errD,
                            errG,
                            D_x,
                            D_G_z1,
                            D_G_z2,
                        )
                    )

                if i % 100 == 0:
                    vutils.save_image(
                        real_cpu, "%s/real_samples.png" % "file", normalize=True
                    )
                    fake = netG(
                        fixed_noise, prev_data_cpu, chord_cpu, batch_size, pitch_range
                    )
                    vutils.save_image(
                        fake.detach(),
                        "%s/fake_samples_epoch_%03d.png" % ("file", epoch),
                        normalize=True,
                    )

            average_lossD = sum_lossD / len(train_loader.dataset)
            average_lossG = sum_lossG / len(train_loader.dataset)
            average_D_x = sum_D_x / len(train_loader.dataset)
            average_D_G_z = sum_D_G_z / len(train_loader.dataset)

            lossD_list.append(average_lossD)
            lossG_list.append(average_lossG)
            D_x_list.append(average_D_x)
            D_G_z_list.append(average_D_G_z)

            print(
                "==> Epoch: {} Average lossD: {:.10f} average_lossG: {:.10f},average D(x): {:.10f},average D(G(z)): {:.10f} ".format(
                    epoch, average_lossD, average_lossG, average_D_x, average_D_G_z
                )
            )

        # save the weights
        np.save("lossD_list.npy", lossD_list)
        np.save("lossG_list.npy", lossG_list)
        np.save("lossD_list_all.npy", lossD_list_all)
        np.save("lossG_list_all.npy", lossG_list_all)
        np.save("D_x_list.npy", D_x_list)
        np.save("D_G_z_list.npy", D_G_z_list)

        # do checkpointing
        torch.save(netG.state_dict(), "%s/netG_epoch_%d.pth" % ("../models", epoch))
        torch.save(netD.state_dict(), "%s/netD_epoch_%d.pth" % ("../models", epoch))

    # This section is responsible for visualizing the training progress by creating a plot of the loss values over time.
    if is_draw == 1:
        lossD_print = np.load("lossD_list.npy")
        lossG_print = np.load("lossG_list.npy")
        length = lossG_print.shape[0]

        x = np.linspace(0, length - 1, length)
        x = np.asarray(x)
        plt.figure()
        plt.plot(x, lossD_print, label=" lossD", linewidth=1.5)
        plt.plot(x, lossG_print, label=" lossG", linewidth=1.5)

        plt.legend(loc="upper right")
        plt.xlabel("data")
        plt.ylabel("loss")
        plt.savefig(
            "where you want to save/lr=" + str(lr) + "_epoch=" + str(epochs) + ".png"
        )
    # This section is used to generate new music samples after the model has been trained.
    if is_sample == 1:
        batch_size = 8
        nz = 100
        n_bars = 7
        X_te = np.load("your testing x")
        prev_X_te = np.load("your testing prev x")
        prev_X_te = prev_X_te[:, :, check_range_st:check_range_ed, :]
        y_te = np.load("yourd chord")

        test_iter = get_dataloader(X_te, prev_X_te, y_te)
        kwargs = {"num_workers": 4, "pin_memory": True}  # if args.cuda else {}
        test_loader = DataLoader(
            test_iter, batch_size=batch_size, shuffle=False, **kwargs
        )

        netG = sample_generator()
        netG.load_state_dict(torch.load("your model"))

        output_songs = []
        output_chords = []
        for i, (data, prev_data, chord) in enumerate(test_loader, 0):
            list_song = []
            first_bar = data[0].view(1, 1, 16, 128)
            list_song.append(first_bar)

            list_chord = []
            first_chord = chord[0].view(1, 13).numpy()
            list_chord.append(first_chord)
            noise = torch.randn(batch_size, nz)

            for bar in range(n_bars):
                z = noise[bar].view(1, nz)
                y = chord[bar].view(1, 13)
                if bar == 0:
                    prev = data[0].view(1, 1, 16, 128)
                else:
                    prev = list_song[bar - 1].view(1, 1, 16, 128)
                sample = netG(z, prev, y, 1, pitch_range)
                list_song.append(sample)
                list_chord.append(y.numpy())

            print("num of output_songs: {}".format(len(output_songs)))
            output_songs.append(list_song)
            output_chords.append(list_chord)
        np.save("output_songs.npy", np.asarray(output_songs))
        np.save("output_chords.npy", np.asarray(output_chords))

        print("creation completed, check out what I make!")


if __name__ == "__main__":
    main()
