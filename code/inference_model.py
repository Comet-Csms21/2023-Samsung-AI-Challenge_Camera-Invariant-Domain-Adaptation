from define_func import *

def InferModel(device, dataloader, model, criterion, optimizer, last_epoch, model_name, backbone_name):
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(dataloader):
            images = images.float().to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1).cpu()
            outputs = torch.argmax(outputs, dim=1).numpy()
            # batch에 존재하는 각 이미지에 대해서 반복
            for pred in outputs:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred) # 이미지로 변환
                pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
                pred = np.array(pred) # 다시 수치로 변환
                # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
                for class_id in range(12):
                    class_mask = (pred == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)
                    else: # 마스크가 존재하지 않는 경우 -1
                        result.append(-1)

    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv(f'./{model_name}_{backbone_name}_{last_epoch+1}.csv', index=False)

def InferModel_val(device, dataset, dataloader, model, criterion, optimizer, last_epoch, model_name, backbone_name):
    if not os.path.exists(f"./{model_name}_{backbone_name}_{last_epoch+1}_val_pred_mask.csv"):
        with torch.no_grad():
            model.eval()
            result = []
            for images in tqdm(dataloader):
                images = images.float().to(device)
                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1).cpu()
                outputs = torch.argmax(outputs, dim=1).numpy()
                # batch에 존재하는 각 이미지에 대해서 반복
                for pred in outputs:
                    pred = pred.astype(np.uint8)
                    pred = Image.fromarray(pred) # 이미지로 변환
                    pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
                    pred = np.array(pred) # 다시 수치로 변환
                    # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
                    for class_id in range(12):
                        class_mask = (pred == class_id).astype(np.uint8)
                        if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                            mask_rle = rle_encode(class_mask)
                            result.append(mask_rle)
                        else: # 마스크가 존재하지 않는 경우 -1
                            result.append(-1)

        #pd.DataFrame(np.array(result), columns=['mask_rle']).to_csv('./DeepLabV3+_drn_30epoch.csv', index=False)
        submit = pd.read_csv("./sample_val_pred_mask_data.csv")
        submit['mask_rle'] = result
        submit.to_csv(f"./{model_name}_{backbone_name}_{last_epoch+1}_val_pred_mask.csv", index=False)

        if not os.path.exists("./val_pred"):
            os.makedirs("./val_pred")

        for i in range(int(len(result)/12)):
            inf_mask = []
            for j in range(i*12, i*12+12):
                inf_mask.append(str(result[j]))
            combine = combined_masks(inf_mask, (540, 960))
            
            image = Image.fromarray(combine.astype(np.uint8))

            # 이미지를 저장
            image.save(f"./val_pred/VALID_SOURCE_{i:03d}.png")

        val_mask_set = CustomDataset(csv_file='./val_source.csv', transform=val_transform, infer=False)
        val_mask_pred_set = CustomDataset(csv_file='./val_pred.csv', transform=val_transform, infer=False)

        miou = 0
        for i in range(len(val_mask_set)):
            _, val_mask = val_mask_set[i]
            _, val_mask_pred = val_mask_pred_set[i]
            #print(val_mask)
            #print("JELL")
            #print(val_mask_pred)
            #iou = monai.metrics.compute_iou(val_mask_pred, val_mask, include_background=False)
            iou = MulticlassJaccardIndex(num_classes=13)
            miou += iou(val_mask, val_mask_pred)
        miou = miou / len(val_mask_set)

        torch.save({
            "model": f"{model_name}_{backbone_name}",
            "mIoU": miou,
            "description": f"{model_name}_{backbone_name}_{last_epoch+1} mIoU값"
            },
            f"./model/{model_name}_{backbone_name}_{last_epoch+1}_val_mIoU.pth"
        )

        print("mIoU 값: ", miou)
        
    else:
        val_mask_set = CustomDataset(csv_file='./val_source.csv', transform=val_transform, infer=False)
        val_mask_pred_set = CustomDataset(csv_file='./val_pred.csv', transform=val_transform, infer=False)

        miou = 0
        for i in range(len(val_mask_set)):
            _, val_mask = val_mask_set[i]
            _, val_mask_pred = val_mask_pred_set[i]
            #print(val_mask)
            #print("JELL")
            #print(val_mask_pred)
            #iou = monai.metrics.compute_iou(val_mask_pred, val_mask, include_background=False)
            iou = MulticlassJaccardIndex(num_classes=13)
            miou += iou(val_mask, val_mask_pred)
        miou = miou / len(val_mask_set)

        torch.save({
            "model": f"{model_name}_{backbone_name}",
            "mIoU": miou,
            "description": f"{model_name}_{backbone_name}_{last_epoch+1} mIoU값"
            },
            f"./model/{model_name}_{backbone_name}_{last_epoch+1}_val_mIoU.pth"
        )

        print("mIoU 값: ", miou)