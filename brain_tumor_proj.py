import os
import glob
import h5py
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

#google drive einbinden
from google.colab import drive
drive.mount('/content/drive')

#gleiche zufälligkeit für reproduzierbarkeit
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

#google drive pfad von daten und neuen pfad für patches (falls noch nicht da)
data_dir="/content/drive/MyDrive/MaschinellesSehen_Projekt"
save_dir=os.path.join(data_dir,"patches")
os.makedirs(save_dir, exist_ok=True)

#alle h5 files suchen
h5_files=glob.glob(os.path.join(data_dir,"*.h5"))
print("anzahl der h5 files:",len(h5_files))
print("ersten 5 files:",h5_files[:5])

#extrahieren von bildpatches mit zugehörigen masken: 50% krebspatches (>1% krebsfläche) und 50% hintergrund, maximal 2500 patches jeweils
def extract_patches(file_list,patch_size=96,tumor_threshold=0.01,max_patches_per_class=2500):
    #liste für tumorpatches
    tumor_patches=[]
    #liste für backgroundpatches
    bg_patches=[]

    for file_path in file_list:
        #stoppen wenn genug patches (mit abbruchkriterium/max matches)
        if len(tumor_patches)>=max_patches_per_class and len(bg_patches)>=max_patches_per_class:
            break

        #h5 datein als read öffnen
        with h5py.File(file_path, 'r') as f:
            #nur T1 images
            img=f['image'][:, :, 0]
            #alle maskenkanäle
            mask=f['mask'][:, :, :]  
            #zusammenfügen mit maximalwert für gleichmäßige 2d maske
            mask=mask.max(axis=2) 

        #image normalisieren auf wert zwischen 0 und 1
        img=(img-img.min())/(img.max()-img.min()+1e-8)
        #binäre maske
        mask=(mask > 0).astype(np.float32)

        height, width=img.shape
        ps = patch_size

        #window über bild sliden
        for y in range(0,height-ps,ps//2):
            for x in range(0,width-ps,ps//2):

                img_patch=img[y:y+ps, x:x+ps]
                mask_patch=mask[y:y+ps, x:x+ps]
                
                #tumoranteil im patch berechnen
                tumor_ratio=mask_patch.sum()/(ps*ps)

                #patch speichern je nach class
                if tumor_ratio>tumor_threshold:
                    if len(tumor_patches)<max_patches_per_class:
                        tumor_patches.append((img_patch,mask_patch))
                else:
                    if len(bg_patches)<max_patches_per_class:
                        bg_patches.append((img_patch,mask_patch))

                #stoppen wenn beide classes voll
                if len(tumor_patches)>=max_patches_per_class and len(bg_patches) >= max_patches_per_class:
                    break
            if len(tumor_patches)>=max_patches_per_class and len(bg_patches) >= max_patches_per_class:
                break

    #tumor und backgroundpatches zusammenfügen und mischen
    combined=tumor_patches+bg_patches
    random.shuffle(combined)

    #bild und maske trennen
    patches_img,patches_mask=zip(*combined)

    #als np speichern
    np.save(os.path.join(save_dir,"patches_img.npy"), np.array(patches_img, dtype=np.float32))
    np.save(os.path.join(save_dir,"patches_mask.npy"), np.array(patches_mask, dtype=np.float32))
    print(f"gespeicherte {len(patches_img)} patches in {save_dir}")

#patches extrahieren ausführen
extract_patches(h5_files)

#dataset klasse, lädt gespeicherte bild und maskenpatches
class PatchDataset(Dataset):
    def __init__(self, img_path, mask_path):
        self.imgs=np.load(img_path)
        self.masks=np.load(mask_path)

    #wie viele patches
    def __len__(self):
        return len(self.imgs)

    #gibt patches mit index zurück
    def __getitem__(self, idx):
        img=torch.from_numpy(self.imgs[idx]).unsqueeze(0)
        mask=torch.from_numpy(self.masks[idx]).unsqueeze(0)
        return img,mask

# 80% training/20% validation splitting der patches
all_img_path=os.path.join(save_dir,"patches_img.npy")
all_mask_path=os.path.join(save_dir,"patches_mask.npy")
all_imgs=np.load(all_img_path)
all_masks=np.load(all_mask_path)

n_total=len(all_imgs)
indices=list(range(n_total))
random.shuffle(indices)

split=int(0.8*n_total)
train_idx=indices[:split]
val_idx=indices[split:]

#spiechern
np.save(os.path.join(save_dir,"train_img.npy"),all_imgs[train_idx])
np.save(os.path.join(save_dir,"train_mask.npy"),all_masks[train_idx])
np.save(os.path.join(save_dir,"val_img.npy"),all_imgs[val_idx])
np.save(os.path.join(save_dir,"val_mask.npy"),all_masks[val_idx])

#dataset und loader
train_ds=PatchDataset(os.path.join(save_dir,"train_img.npy"),
                      os.path.join(save_dir,"train_mask.npy"))
val_ds=PatchDataset(os.path.join(save_dir,"val_img.npy"),
                    os.path.join(save_dir,"val_mask.npy"))

train_loader=DataLoader(train_ds,batch_size=8,shuffle=True)
val_loader=DataLoader(val_ds,batch_size=8,shuffle=False)

#unet modell
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(out_ch,out_ch,3,padding=1)
        self.relu2=nn.ReLU()

    def forward(self, x):
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        #encoder
        self.enc1=DoubleConv(1,32)
        self.enc2=DoubleConv(32,64)
        self.pool=nn.MaxPool2d(2)
        #bottleneck
        self.mid=DoubleConv(64,128)
        #decoder
        self.up2=nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec2=DoubleConv(128,64)
        self.up1=nn.ConvTranspose2d(64,32,2,stride=2)
        self.dec1=DoubleConv(64,32)
        self.out=nn.Conv2d(32,1,1)

    def forward(self, x):
        e1=self.enc1(x)
        e2=self.enc2(self.pool(e1))
        m=self.mid(self.pool(e2))
        d2=self.up2(m)
        d2=self.dec2(torch.cat([d2,e2],dim=1))
        d1=self.up1(d2)
        d1=self.dec1(torch.cat([d1,e1],dim=1))
        return self.out(d1)

#loss funktionen
def dice_score(pred,target,eps=1e-6):
    pred=(torch.sigmoid(pred)>0.5).float()
    inter=(pred*target).sum()
    union=pred.sum(dim=()) + target.sum()
    return ((2*inter+eps)/(union+eps)).mean()

def dice_loss(pred,target,eps=1e-6):
    pred_sig=torch.sigmoid(pred)
    inter=(pred_sig*target).sum()
    dice=(2*inter+eps)/(pred_sig.sum()+target.sum() + eps)
    return 1-dice

def loss_bce_dice(pred,target):
    bce=nn.BCEWithLogitsLoss()(pred,target)
    d=dice_loss(pred,target)
    return bce+d

#training
def train_model(loss_function,loss_name,epochs=45):
    device=torch.device("cuda" if torch.cuda.is_available()else "cpu")
    print("using:", device)

    model = UNet().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

    train_losses=[] 
    val_losses=[] 
    val_dices=[]

    for epoch in range(epochs):
        #training
        model.train()
        running_loss=0
        for imgs, masks in train_loader:
            imgs=imgs.to(device)
            masks=masks.to(device)

            optimizer.zero_grad()
            preds=model(imgs)

            loss=loss_function(preds, masks)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

        train_loss=running_loss/len(train_loader)
        train_losses.append(train_loss)

        #validation
        model.eval()
        val_loss=0
        dice_total=0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs=imgs.to(device)
                masks=masks.to(device)

                preds=model(imgs)
                loss=loss_function(preds,masks)

                val_loss+=loss.item()
                dice_total+=dice_score(preds,masks).item()
        
        val_loss/=len(val_loader)
        val_dice=dice_total/len(val_loader)

        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(f"{loss_name} | Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f}")

        #overlay nach jeder epoche mit zufälligem tumor patch aus validation
        with torch.no_grad():
            img_patch=None
            gt_patch=None
            pred_patch=None

            #durch alle in validation
            for imgs, masks in val_loader:
                tumor_pixels_per_patch=masks.view(masks.size(0),-1).sum(dim=1)
                #indey aller patches mit tumor (>0 pixel)
                tumor_indices=(tumor_pixels_per_patch>0).nonzero(as_tuple=True)[0]
                if len(tumor_indices)>0:
                    #zufälliges tumorpatch auswählen
                    idx = random.choice(tumor_indices).item()
                    #bild und maske vorbereiten
                    img_patch=imgs[idx].cpu().squeeze().numpy()
                    gt_patch=masks[idx].cpu().squeeze().numpy()
                    #modellvorhersage für dieses patch
                    pred=model(imgs[idx:idx+1].to(device))

                    #sigmoid und threshold für binäre maske
                    pred_patch=(torch.sigmoid(pred) > 0.5).float()
                    pred_patch=pred_patch.cpu().squeeze().numpy()
                    #nur ein patch
                    break

            #plotting
            if img_patch is not None:
                plt.figure(figsize=(15,5))
                #t1 bild
                plt.subplot(1,3,1)
                plt.imshow(img_patch, cmap="gray")
                plt.title("T1 Patch")
                plt.axis("off")

                #ground truth overlay
                plt.subplot(1,3,2)
                #rgba
                gt_color = np.zeros((*gt_patch.shape, 4))
                #in grün
                gt_color[...,1] = 1.0
                #0,5 transparenz
                gt_color[...,3] = gt_patch*0.5
                plt.imshow(img_patch, cmap="gray")
                plt.imshow(gt_color)
                plt.title("Ground Truth Overlay")
                plt.axis("off")

                #prediction overlay
                plt.subplot(1,3,3)
                pr_color = np.zeros((*pred_patch.shape, 4))
                #in rot
                pr_color[...,0] = 1.0
                pr_color[...,3] = pred_patch*0.5
                plt.imshow(img_patch, cmap="gray")
                plt.imshow(pr_color)
                plt.title("Prediction Overlay")
                plt.axis("off")
                plt.show()

    return model, train_losses, val_losses, val_dices

#training starten
results={}
print("\nTraining mit Dice Loss allein")
results["Dice"]=train_model(dice_loss, "Dice")

print("\nTraining mit BCE + Dice Loss")
results["BCE+Dice"]=train_model(loss_bce_dice,"BCE+Dice", epochs=45)

# --------------------------
# Performance nach Training
for name in results:
    model, train_l, val_l, val_d = results[name]

    plt.figure()
    plt.plot(train_l,label="Train Loss")
    plt.plot(val_l,label="Val Loss")
    plt.title(f"Loss Verlauf - {name}")
    plt.legend()
    plt.show()

#vergleich mit dice für beide loss funktionen
plt.plot(results["Dice"][3], label="Dice Loss")
plt.plot(results["BCE+Dice"][3], label="BCE+Dice Loss")

plt.title("Validation Dice Vergleich")
plt.xlabel("Epochen")
plt.ylabel("Dice-Score")
plt.ylim(0, 1)  # Y-Achse von 0 bis 1
plt.legend()
plt.grid(True)
plt.show()