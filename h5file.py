import h5py
import numpy as np
import matplotlib.pyplot as plt

#daten laden, öffnen als read
filename="BraTS2020_training_data/content/data/volume_1_slice_90.h5"

with h5py.File(filename,'r') as f:
    #bildarray in (240, 240, 4 (mrt-sequenzen)) ['T1', 'T1CE', 'T2', 'FLAIR']
    img_arr=f['image'][:]  
    #maskenarray in (240, 240, 3 (tumorklassen))
    img_mask=f['mask'][:] 
    print("image:",img_arr.shape)
    print("mask:",img_mask.shape)

#maske auf eine dimension reduzieren
if img_mask.ndim==3 and img_mask.shape[2]==3:
    #nue maximalwert über alle kanäle nehmen
    mask_single=img_mask.max(axis=2)

#binäre Maske (bool array) mit tumorregion als true
mask_bin=mask_single>0 

#plots: eine zeile mit drei plots
fig,axes=plt.subplots(1,3,figsize=(16, 6))

#t1-bild im ersten kanal
t1_img=img_arr[:,:,0] 
#normalisierung auf 0 bis 255
min_val=t1_img.min()
max_val=t1_img.max()
t1_norm=((t1_img-min_val)*255/(max_val-min_val)).astype('uint8')

axes[0].imshow(t1_norm,cmap='gray')
axes[0].set_title("T1 Slice")
axes[0].axis('off')

#maske
mask_rgb=np.zeros((*mask_single.shape,3),dtype=np.uint8)
#grüne füllung für tumorregion
mask_rgb[mask_single>0]=(0,255,0)

axes[1].imshow(mask_rgb)
axes[1].set_title("Binary Mask (Green)")
axes[1].axis('off')

overlay=np.zeros((*mask_single.shape, 4),dtype=np.float32)
#grünkanal
overlay[..., 1]=1.0       
#transparent 0,5  
overlay[..., 3]=mask_bin*0.5 

axes[2].imshow(t1_norm, cmap="gray")
axes[2].imshow(overlay)
axes[2].set_title("Ground Truth Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.show()