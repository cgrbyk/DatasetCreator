import cv2
import numpy as np
import os
import random as rd
from generate_xml import write_xml

tl_list = []
br_list = []
object_list = []
dataset_folder_path='C:\\Users\\cgrbyk\\Desktop\\Dataset\\'
savedir=dataset_folder_path+'annotations'
image_folder=dataset_folder_path+'FinalImages'

Backlist = os.listdir(dataset_folder_path+'BacgroundImages')#Arka plan resimlerinin listesi
BackgroundCount = len(Backlist)
i=0
for n, CropfileName in enumerate(os.scandir(dataset_folder_path+'CroppedImages')): #Kesilmiş resimler içerisindeki her bir klasör için işlem tekrarlanacak
    Croplist = os.listdir(dataset_folder_path+'CroppedImages\\'+CropfileName.name) #Sıradaki klasörün içerisindeki resimlerin listesi
    CroppedImageCount = len(Croplist)
    while(i<CroppedImageCount): # bütün kesilmiş resimler tamamlanana kadar işlem devam ediyor.
        Croppedimage = cv2.imread(dataset_folder_path+'CroppedImages\\'+CropfileName.name+'\\'+Croplist[i])
        Backgroundimage=cv2.imread(dataset_folder_path+'BacgroundImages\\'+Backlist[rd.randint(0, BackgroundCount)])
        img2=Croppedimage#kesilmiş olan resmi arka plan resmi üzerisine koyma işlemi başlangıçı
        img1=Backgroundimage
        rows, cols, channels = img2.shape
        roi = img1[0:rows,0:cols]
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        try: #eğer bir resim okunamıyorsa burada hata veriyor resmin adını yazdırıyoruz ve datasetten çıkartıyoruz.
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        except:
            print(CropfileName.name+'\\'+Croplist[i])
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
        dst = cv2.add(img1_bg, img2_fg)
        rndRow=rd.randint(0,(Backgroundimage.shape[0]-Croppedimage.shape[0])) #Arka plan üzerine koyulan resim için rastgele bir satır belirleniyor ancak bu sayı merkezi temsil ettiği için arkaplanın maksimum yüksekliğinden yerleştirilen resmin yüksekliğinin yarısı çıkartılıyor.
        rndCol = rd.randint(0, (Backgroundimage.shape[1] - Croppedimage.shape[1]))#yukarıdaki işlem sütun için tekrarlanıyor.
        img1[rndRow:rows+rndRow,rndCol:cols+rndCol] = dst #Kesilmiş resmin arkaplan üzerine koyulması tamamlandı.
        #cv2.imshow('res', img1)
        cv2.imwrite(dataset_folder_path+'FinalImages\\{:06}.png'.format(i),img1) #resimler altı haneli bir formatta kaydediliyor.
        br_list.append((int(rndRow+(Croppedimage.shape[0]/2)), (int(rndCol+(Croppedimage.shape[1]/2))))) #rastgele belirlenen merkez noktası üzerinden aşağı sağ(bottom right br) ve yukarı sol (top left tl) noktaları hesaplanıyor
        tl_list.append((int(rndRow-(Croppedimage.shape[0]/2)), (int(rndCol-(Croppedimage.shape[1]/2))))) #hesaplanan bu noktalar xml oluştururken kullanılacak
        object_list.append(CropfileName.name)#Klasör ismi aynı zaman içerisindeki resimlerin eğitim türü adı oluyor
        write_xml(image_folder, dataset_folder_path+'FinalImages\\{:06}.png'.format(i),'{:06}.png'.format(i), object_list, tl_list, br_list, savedir) #xml oluşturma komutu. Bu komut ile annotation dosyaları oluşuyor.
        tl_list.clear() #listeler sıfırlanıyor
        br_list.clear()
        object_list.clear()
        i=i+1 #isimlendirmek için kullanılan sayaç bir artırılıyor.
