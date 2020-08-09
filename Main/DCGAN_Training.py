from Main.DCGAN import *

# Training
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")

# Using for-each epoch
for epoch in range(num_epochs):
    # for each batch
    for i, data in enumerate(train_loader, 0):
        # Update 0
        # Train with All-Real-Batch.
        """
            Zero-gradient of netD.
        """
        netD.zero_grad()
        real_cpu = data[0].to(DEVICE)
        b_size = real_cpu.size(0)
        label = torch.full(
            (b_size,), real_label,
            device=DEVICE
        )

        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with All-Fake-Batch
        noise = torch.randn(
            b_size, nz, 1, 1,
            device=DEVICE
        )
        fake = netG(noise)
        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizerD.step()

        """
            Zero-gradient of netG.
        """
        # Update G
        netG.zero_grad()
        label.fill_(real_label)

        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        if i % 50 == 0:
            print("[%d %d] [%d %d]\t "
                  "Loss_D : %.4f\t Loss_G : %.4f\t"
                  "D(x) : %.4f\t D(G(z)) : %.4f / %.4f" % (
                      epoch, num_epochs, i, len(train_loader),
                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2
                  ))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if(iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(train_loader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1


plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
