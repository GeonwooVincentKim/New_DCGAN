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
        netD.zero_grad()
        real_cpu = data[0].to(DEVICE)
        b_size = real_cpu.size(0)
        label = torch.full(
            (b_size, ), real_label,
            device=DEVICE
        )
