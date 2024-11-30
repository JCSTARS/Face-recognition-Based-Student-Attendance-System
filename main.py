import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# 1. 初始化模型
mtcnn = MTCNN(keep_all=True, device='cuda')
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
# 2. 加载目标人脸库
def load_face_embeddings(folder_path):
    embeddings = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)[:, :, ::-1]  # Convert BGR to RGB
            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                cropped_face = mtcnn(img, save_path=None)
                cropped_face = cropped_face.squeeze(0)
                embedding = resnet(cropped_face.unsqueeze(0).cuda()).detach().cpu().numpy()
                embeddings[filename] = embedding
    return embeddings

target_faces = load_face_embeddings('target_faces')

# 3. 处理输入图片
def recognize_faces(input_image, target_embeddings):
    img = cv2.imread(input_image)[:, :, ::-1]
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return []

    faces = mtcnn(img, save_path=None)
    input_embeddings = [resnet(face.unsqueeze(0).cuda()).detach().cpu().numpy() for face in faces]

    results = []
    for face_emb in input_embeddings:
        similarities = {}
        for name, target_emb in target_embeddings.items():
            similarity = cosine_similarity(face_emb, target_emb)[0][0]
            probabilities = 1 / (1 + np.exp(-similarity))  # 转化为概率
            similarities[name] = probabilities
        results.append(similarities)
    return results

image_path = 'group_photo/photo2.jpg'
results = recognize_faces(image_path, target_faces)

# 打印结果
for i, res in enumerate(results):
    print(f"Face {i + 1}:")
    for target, prob in res.items():
        print(f" - {target}: {prob:.2f}")
        