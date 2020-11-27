from torchvision import transforms
import albumentations as A

""" mendefinisikan tranformasi data input """
# transform = transforms.Compose([
#     transforms.Resize(128,171),                     #mengubah ukuran image menjadi 128 x 171
#     transforms.CenterCrop(112),                     #melakukan center crop to the frame menjadi 112 x 112
#     transforms.Normalize(                         #melakukan normalisasi data sesuai pretrained ResNet 18 3D
#                 mean = [0.43216, 0.394666, 0.37645],
#                 std = [0.22803, 0.22145, 0.216989]),
#     transforms.ToTensor()
# ])

transform = A.Compose([
    A.Resize(128,171, always_apply=True),                     #mengubah ukuran image menjadi 128 x 171
    A.CenterCrop(112,112, always_apply=True),                     #melakukan center crop to the frame menjadi 112 x 112
    A.Normalize(                         #melakukan normalisasi data sesuai pretrained ResNet 18 3D
                mean = [0.43216, 0.394666, 0.37645],
                std = [0.22803, 0.22145, 0.216989], always_apply=True)
    # transforms.ToTensor()
])

""" membaca class names dari label.txt """
with open('labels.txt', 'r') as f:
    class_names = f.readlines()
    f.close()

""" menghitung jumlah label yang digunakan atau disediakan pada label.txt"""
count = 0
for i in class_names:
    count += 1;
print("jumlah label : ", count)