"""
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
app.config['RESULT_FOLDER'] = './static/results/'
app.config['TARGET_FOLDER'] = './target_faces/'

# 初始化模型
mtcnn = MTCNN(keep_all=True, device='cuda')
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

# 加载目标人脸
def load_target_faces(folder_path):
    embeddings = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                cropped_face = mtcnn(img, save_path=None)
                embedding = resnet(cropped_face.cuda()).detach().cpu().numpy()
                embeddings[filename] = embedding
    return embeddings

target_faces = load_target_faces(app.config['TARGET_FOLDER'])

# 识别图片中的人脸
def recognize_faces(input_image, target_embeddings):
    img = cv2.imread(input_image)[:, :, ::-1]
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return [], None

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
    return results, boxes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        if not uploaded_files:
            return redirect(url_for('index'))
        
        for file in uploaded_files:
            if file:
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # 识别人脸
                results, boxes = recognize_faces(filepath, target_faces)

                img = cv2.imread(filepath)
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(b) for b in box]
                        
                    max_confidence_label = max(results[i], key=results[i].get)
                    max_confidence_value = results[i][max_confidence_label]
                    max_confidence_label = max_confidence_label.split('.')[0]
                    if max_confidence_value < 0.6:
                        continue
                    label = f"{max_confidence_label}: {max_confidence_value:.2f}"
        
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                
                filename="res-img.jpg"
                result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
                cv2.imwrite(result_path, img)

            return redirect(url_for('result', filename=filename))
    return render_template('index.html')

@app.route('/<filename>')
def result(filename):
    result_path = os.path.join('static/results', filename)
    return render_template('result.html', result_image=result_path)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)
"""
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
app.config['RESULT_FOLDER'] = './static/results/'
app.config['TARGET_FOLDER'] = './target_faces/'

# 初始化模型
mtcnn = MTCNN(keep_all=True, device='cuda')
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

# 加载目标人脸
def load_target_faces(folder_path):
    embeddings = {}
    namelist=[]
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
            boxes, _ = mtcnn.detect(img)
            if boxes is not None:
                cropped_face = mtcnn(img, save_path=None)
                embedding = resnet(cropped_face.cuda()).detach().cpu().numpy()
                embeddings[filename] = embedding
                namelist.append(filename)
    return embeddings, namelist

target_faces, namelist = load_target_faces(app.config['TARGET_FOLDER'])

# 识别图片中的人脸
def recognize_faces(input_image, target_embeddings):
    img = cv2.imread(input_image)[:, :, ::-1]
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return [], None

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
    return results, boxes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('files')
        if not uploaded_files:
            return redirect(url_for('index'))

        label_counts = {}  # 统计每个标签的数量
        uploaded_image_paths = []  # 保存上传图片路径
        result_images = []

        for fname in namelist:
            label_counts[fname.split('.')[0]] = "Abscent"
        for file in uploaded_files:
            if file:
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_image_paths.append(filepath)
                target_width = 1024
                img=cv2.imread(filepath)
                scale_ratio = target_width / img.shape[1]
                new_height = int(img.shape[0] * scale_ratio)
                img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(filepath, img)
                # 识别人脸
                results, boxes = recognize_faces(filepath, target_faces)

                # 标注图片
                img = cv2.imread(filepath)
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(b) for b in box]
                    # 初始化最大置信度和对应标签
                    max_confidence = 0
                    max_label = 'Unknown'
                    
                    for k, v in results[i].items():
                        if v > 0.6:  # 阈值为0.6
                            if v > max_confidence:  # 找到最大的置信度
                                max_confidence = v
                                max_label = k.split('.')[0]  # 提取目标人脸标签
                    
                    #label_counts[max_label] = label_counts.get(max_label, 0) + 1
                    if max_label != "Unknown":
                        label_counts[max_label] = "Attend"
                    else:
                        continue
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, max_label+":"+"{:.2f}".format(max_confidence), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 保存标注图片
                result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
                cv2.imwrite(result_path, img)
                result_images.append(result_path)

        return render_template(
            'result.html',
            uploaded_images=uploaded_image_paths,
            result_images=result_images,
            label_counts=label_counts
        )
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploads(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True)
