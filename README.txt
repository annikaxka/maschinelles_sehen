Dieses Projekt implementiert eine binäre Tumorsegmentierung auf Grundlage des BraTS2020-Datensatzes mithilfe von Deep Learning.
Ziel ist es, Tumorregionen in MRT-T1-Slices automatisch zu erkennen.

Projektidee:
Die manuelle Segmentierung von Gehirntumoren in MRT-Daten ist sehr zeitaufwändig und subjektiv. Ziel dieses Projekts ist eine automatisierte, präzise Segmentierung von Gehirntumoren in MRT-T1-Slices mithilfe von Deep Learning. Anhand der MRT-Slices und zugehörigen Tumormasken des BraTS2020-Datensatzes soll ein einfaches, neuronales 2D-UNet mit klassenbalancierten Tumor- und Backgroundpatches trainiert werden, das die Tumorregionen zuverlässig erkennt und eine entsprechende binäre Maske erzeugt.

Datengrundlage:
BraTS2020-Datensatz
https://www.kaggle.com/datasets/awsaf49/brats2020-training-data 
369 Gehirn MRTs mit einem Tumor, je 154 Slices pro Scan und zugehörigen Gehirntumormasken

Files im Projekt:
Poster_brain_tumor_proj.pdf
    -wissenschaftliches Poster zum Projekt
h5file.py 
    -zur Visualisierung und Analyse einer einzelnen H5-Datei (einzelnes Slice eines Volumes)
output_h5file.png
    -Output der h5file.py Datei
brain_tumor_proj.py
    -implementiert die vollständige Pipeline zur binären Tumorsegmentierung von MRT-T1-Bildern
     mithilfe eines UNet Modells
output_each_epoch_brain_tumor_proj.pdf
    -Output jeder Epoche mit verwendeter Loss Metrik, Anzahl der Epoche, Train Loss, Val Loss,
     Dice Score und Stand des aktuellen Netzes mit Visualisierung von random Validation Tumor
    Patch, Ground Truth und Predicted Mask
output_plots_loss_dice_brain_tumor_proj.pdf
    -finale Plots von Train und Val Loss über die 45 Epochen für Dice Loss und Dice + BCE Loss
    -Vergleich von Dice Score über 45 Epochen für Training mit Dice Loss und Dice + BCE Loss

relevante Papers und Referenzen:
    - zu UNet-Architekturen, Patch-basierter Approach, Dice Evaluation
        Lefkovits,S.,Kovács,A.,&Szabó,T.(2022).UNet architecture variants for brain tumor
        segmentation. Acta Universitatis Sapientiae, Informatica, 14(1),23–38.
        DOI:10.2478/ausi-2022-0004
    - zu Klassenbalanciert, 2D Patch-basiertes CNN mit Sliding Window 
        Kao P-Y, Shailja S, Jiang J, Zhang A, Khan A, Chen JW and Manjunath BS (2020) Improving
        Patch-Based Convolutional Neural Networks for MRI Brain Tumor Segmentation by
        Leveraging Location Information. Front. Neurosci. 13:1449.
        DOI: 10.3389/fnins.2019.01449
    - zu Dice Loss und Kombi mit BCE Loss
        Sudre,C.H.,Li,W.,Vercauteren,T.,Ourselin,S.,&JorgeCardoso,M (2017).Generalised Dice
        overlap as a deep learning loss function for highly unbalanced segmentations. In Deep
        Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support
        (pp.240–248).Springer.
        DOI:10.1007/978-3-319-67558-9_28
    - Metriken Dice-Loss, BCE-Loss, Dice-Score
        Jadon,S (2020). A survey of loss functions for semantic segmentation. 2020 IEEE
        Conference on Computational Intelligence in Bioinformatics and Computational Biology
        (CIBCB), 1–7.
        DOI:10.1109/CIBCB48159.2020.9277638

Python-Dateien und Vorgehensweise:
h5file.py
    - die H5-Datei aus dem BraTS2020 Datensatz enthälgt
        -image (240,240,4)
            =das MRT-Bild als T1, T1CE, T2 und FLAIR
        -mask (240,240,3)
            =die Tumormaske mit drei Tumorklassen
    - die Tumormaske wird auf eine binäre Maske reduziert
    - es werden insgesamt 3 Darstellungen erzeugt
        - T1 Slice in Graustufen
        - binäre Maske in grün
        - und ein Overlay aus T1 MRT Slice und semitransparente Tumormaske

brain_tumor_proj.py - Durchführung in Google Colab
    - die 240x240 T1 Slices und deren Masken werden in kleinere Patches aufgeteilt
        - Patchgröße: 96x96
        - Sliding Window mit Schrittweite patch_size // 2
        - Klassenbalanciert 50%/50% mit max. 2500 Tumor Patches und 2500 Backgrounf Patches
                - Tumor-Patch mit >1% Tumorfläche im Bild
                - gespeichert als:
                    - patches_img.npy
                    - patches_mask.npy
    -80% Training, 20% Validierungsdaten
        -gespeichert als:
            - train_img.npy
            - train_mask.npy
            - val_img.npy
            - val_mask.npy
    - PatchDataset Klasse
        - lädt numpy Dateien
        - konvertiert sie zu torch tensoren
        - fügt kanaldimension hinzu
        - batch_size =8
    -UNet Modell
        -Architektur
            - Input: 1x96x96
            Encoder - downsampling - erkennt Strukturen/globale Infos:
                - Double Conv mit ReLU 1 -> 32
                - MaxPool, Halbierung der Bildgröße
                - DoubleConv mit ReLU 32 -> 64
                - MaxPool, Halbierung der Bildgröße
            Bottleneck - komplexeste Features:
                - DoubleConv mit ReLU 64 -> 128
            Decoder - upsampling:
                - UpConv 128 -> 64
                - Skip Connection (mit unterem Encoder für Details aus Encoder)
                - DoubleConv mit ReLU 128 -> 64
                - UpConv 64 -> 32
                - Skip Connection (mit oberem Encoder für Details aus Encoder)
                - DoubleConv mit ReLU 64 -> 32
            Output: 1x1 Conv - Pixelweise Klassifikation mit Tumorwahrscheinlichkeit
        -Loss Funktionen 
            -Dice-Loss
            -und Dice und BCE Loss
        -Training
            -Optimizer: Adam (Adaptive Moment Estimation)
                - besonders gut für kleine Zielregionen und med. Bilder
                - robuster als SDG 
            -Lernrate: 1e-4
            -Epochen:45
            - pro Epoche:
                -Training (4000 Patches mit Minibatches von 8 Patches)
                    - 500 Minibatch-Updates der Gewichte
                -Validierung
                -Berechnung:
                    - durchschnittlicher Training Loss
                    - durchschnittlicher Validation Loss
                    -durchschnittlicher Dice-Score
                    -Visualiserung mit zufälligem Tumor-Patch
                        -T1
                        -Ground Truth Overlay (grün)
                        -Prediction Overlay (rot)
    -Evaluierung
    -Plotten von Training und Validation Loss für Dice Loss und BCE + Dice Loss 
    -Vergleich der Dice Scores von Dice Loss und BCE + Dice Loss 
