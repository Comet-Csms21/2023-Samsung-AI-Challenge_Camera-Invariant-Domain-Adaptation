from define_func import *
from modeling.discriminator import FCDiscriminator

def TrainModel(device, dataloader, target_dataloader, model, criterion, optimizer, scheduler, epochs, model_name, backbone_name, last_epoch=-1):
    print(device)

    model_folder = "./model"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    input_size = np.array([2048, 1024])
    input_size_target = np.array([1920, 1080])
    size = np.array([224, 384])

    model_D = FCDiscriminator(num_classes=13).to(device)

    #optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.001, betas=(0.9, 0.99))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.0001)

    if last_epoch != -1:
        checkpoint_D = torch.load(f'./model/{model_name}_{backbone_name}_checkpoint-{last_epoch + 1:04d}.pth')
        model_D.load_state_dict(checkpoint_D["model_D_state_dict"])
        optimizer_D.load_state_dict(checkpoint_D["optimizer_D_state_dict"])

    bce_loss = torch.nn.BCEWithLogitsLoss()

    #interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    #interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # training loop
    for epoch in range(last_epoch + 1, epochs):
        model.train()
        model_D.train()

        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_D_value = 0

        optimizer.zero_grad()
        optimizer_D.zero_grad()

        for _, data in enumerate(zip_longest(tqdm(dataloader), target_dataloader)):
            data1, data2 = data
            images, masks = data1
            target_images = data2

            # train G
            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train with source
            images = images.float().to(device)
            masks = masks.long().to(device)

            pred = model(images)
            #pred = interp(pred)

            loss_seg = criterion(pred, masks.unsqueeze(1))
            loss = loss_seg

            loss.backward()

            loss_seg_value += loss_seg.item()

            # train with target
            target_images = target_images.float().to(device)
            
            pred_target = model(target_images)
            #pred_target = interp_target(pred_target)

            #source_label = np.random.randint(0, 13, size=[pred_target.data.size()[0], 1, 7, 12])
            #source_label = torch.from_numpy(source_label).to(device)
            
            D_out = model_D(F.softmax(pred_target, 1))

            loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

            loss = loss_adv_target * 0.001

            loss.backward()

            loss_adv_target_value += loss_adv_target.item()

            # train D
            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            pred = pred.detach()

            D_out = model_D(F.softmax(pred, 1))

            #source_label = np.random.randint(0, 13, size=[pred.data.size()[0], 1, 7, 12])
            #source_label = torch.from_numpy(source_label).to(device)

            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))/2

            loss_D.backward()

            loss_D_value += loss_D.item()

            # train with target
            pred_target = pred_target.detach()

            D_out = model_D(F.softmax(pred_target, 1))

            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))/2

            loss_D.backward()

            loss_D_value += loss_D.item()
        optimizer.step()
        optimizer_D.step()
        #scheduler.step()
        
        print('Epoch: %d, lr: %f, loss_seg_value: %f, loss_adv_target_value: %f, loss_D_value: %f' %(epoch+1, optimizer.param_groups[0]['lr'], loss_seg_value/len(dataloader), loss_adv_target_value/len(dataloader), loss_D_value/len(dataloader)))

        if (epoch + 1) % 5 == 0:
            torch.save({
                "model": f"{model_name}_{backbone_name}",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_D_state_dict": model_D.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "loss_seg_value": loss_seg_value/len(dataloader),
                "loss_adv_target_value": loss_adv_target_value/len(dataloader),
                "loss_D_value": loss_D_value/len(dataloader),
                "description": f"{model_name}_{backbone_name} 체크포인트-{epoch + 1:04d}"
                },
                f"./model/{model_name}_{backbone_name}_checkpoint-{epoch + 1:04d}.pth"
            )
            print("Model Saved!")
        else:
            torch.save({
                "model": "{model_name}_{backbone_name}",
                "epoch": epoch,
                "loss_seg_value": loss_seg_value/len(dataloader),
                "loss_adv_target_value": loss_adv_target_value/len(dataloader),
                "loss_D_value": loss_D_value/len(dataloader),
                "description": f"{model_name}_{backbone_name} 체크포인트-{epoch + 1:04d}"
                },
                f"./model/{model_name}_{backbone_name}_checkpoint-{epoch + 1:04d}.pth"
            )

    model_D.eval()