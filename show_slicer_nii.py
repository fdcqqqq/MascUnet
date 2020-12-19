import SimpleITK as sitk
import skimage.io as io
import matplotlib.pyplot as plt


def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data


def show_img_LGG(ori_img):
    plt.imshow(ori_img[70])

    plt.axis('off')


def show_img_HGG(ori_img):
    plt.imshow(ori_img[105])

    plt.axis('off')


path1 = 'HGG_flair.nii.gz'
path2 = 'HGG_seg.nii.gz'
path3 = 'attention_unet_HGG.nii.gz'
path4 = 'CBAM_unet_HGG.nii.gz'
path5 = 'CBAM_resunet_HGG.nii.gz'
path6 = 'masc_unet_HGG.nii.gz'
path7 = 'parallel_unet_HGG.nii.gz'
path8 = 'unet_attention_HGG.nii.gz'



path9 = 'LGG_flair.nii.gz'
path10 = 'LGG_seg.nii.gz'
path11 = 'attention_unet_LGG.nii.gz'
path12 = 'CBAM_unet_LGG.nii.gz'
path13 = 'CBAM_resunet_LGG.nii.gz'
path14 = 'masc_unet_LGG.nii.gz'
path15 = 'parallel_unet_LGG.nii.gz'
path16 = 'unet_attention_LGG.nii.gz'



data1 = read_img(path1)
data2 = read_img(path2)
data3 = read_img(path3)
data4 = read_img(path4)
data5 = read_img(path5)
data6 = read_img(path6)
data7 = read_img(path7)
data8 = read_img(path8)

data9 = read_img(path9)
data10 = read_img(path10)
data11 = read_img(path11)
data12 = read_img(path12)
data13 = read_img(path13)
data14 = read_img(path14)
data15 = read_img(path15)
data16 = read_img(path16)


plt.show()
plt.figure(figsize=(100, 100), dpi=80)

# plt.suptitle('Result',size=36)

# plt.figure(1)
ax1 = plt.subplot(2, 8, 1)
ax1.set_title('Input Image', size=18)
show_img_HGG(data1)

ax2 = plt.subplot(2, 8, 2)
ax2.set_title('Ground Truth', size=18)
show_img_HGG(data2)

ax3 = plt.subplot(2, 8, 3)
ax3.set_title('Attention Unet', size=18)
show_img_HGG(data3)

ax4 = plt.subplot(2, 8, 4)
ax4.set_title('CBAM Unet', size=18)
show_img_HGG(data4)

ax5 = plt.subplot(2, 8, 5)
ax5.set_title('CBAM ResUnet', size=18)
show_img_HGG(data5)

ax6 = plt.subplot(2, 8, 6)
ax6.set_title('MASC Unet', size=18)
show_img_HGG(data6)
ax7 = plt.subplot(2, 8, 7)
ax7.set_title('Parallel Unet', size=18)
show_img_HGG(data7)
ax8 = plt.subplot(2, 8, 8)
ax8.set_title('ParallelMascUnet(ours)', size=18)
show_img_HGG(data8)

ax9 = plt.subplot(2, 8, 9)
# ax6.set_title('Prediction', size=18)
show_img_LGG(data9)
ax10 = plt.subplot(2, 8, 10)
# ax6.set_title('Prediction', size=18)
show_img_LGG(data10)
ax11 = plt.subplot(2, 8, 11)
# ax6.set_title('Prediction', size=18)
show_img_LGG(data11)
ax12 = plt.subplot(2, 8, 12)
# ax6.set_title('Prediction', size=18)
show_img_LGG(data12)
ax13 = plt.subplot(2, 8, 13)
# ax6.set_title('Prediction', size=18)
show_img_LGG(data13)
ax14 = plt.subplot(2, 8, 14)
# ax6.set_title('Prediction', size=18)
show_img_LGG(data14)
ax15 = plt.subplot(2, 8, 15)
# ax6.set_title('Prediction', size=18)
show_img_LGG(data15)
ax16 = plt.subplot(2, 8, 16)
# ax6.set_title('Prediction', size=18)
show_img_LGG(data16)
plt.tight_layout()

# plt.subplots_adjust(wspace =0.01, hspace =0.01)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=-0.49)
plt.show()
