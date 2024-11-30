import cv2
import os

# 设置lfw数据集的路径和保存人脸的路径
lfw_dataset_path = '/hdd/wxy/scikit_learn_data/lfw_home/lfw_funneled'  # 替换为你的lfw数据集路径
save_faces_path = '/hdd/wxy/scikit_learn_data/lfw_home_haar/lfw_haar'  # 替换为你想要保存人脸的路径

# 加载Haar Cascades人脸检测器
face_cascade = cv2.CascadeClassifier('/hdd/wxy/face_recognition/haarcascade_frontalface_default.xml')  # 替换为你的Haar Cascades文件路径

# 遍历lfw数据集中的每个人脸文件夹
for person in os.listdir(lfw_dataset_path):
    person_path = os.path.join(lfw_dataset_path, person)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                for (x, y, w, h) in faces:
                    face = image[y:y+h, x:x+w]
                    save_folder = os.path.join(save_faces_path, person)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    face = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(os.path.join(save_folder, image_name), face)

print("Face detection and saving complete.")