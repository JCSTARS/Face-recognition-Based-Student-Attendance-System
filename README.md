# Face-recognition-Based-Student-Attendance-System
The system uses MTCNN for face detection and InceptionResnetV1 pre-trained
on the VGGFace2 dataset for face recognition integrates, by extracting embeddings and compares them against a predefined database of
target faces using cosine similarity, with web-interface.
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

Provide step-by-step installation instructions:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JCSTARS/Face-recognition-Based-Student-Attendance-System.git
    ```

2. **Install dependencies:**
   ```bash
   cd Face-recognition-Based-Student-Attendance-System
   pip install -r requirements.txt
   ```
   
3. **Run the system:**
   ```bash
   python app.py
   ```

## Usage
First, upload the images of each person to ``` target-face ``` directory.

After running app.py, the front end should be activate on port 5000 at 127.0.0.1(host), then upload the images and press 'upload' buttom. The time and corresponding face should be annotated and added up then.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](https://github.com/JCSTARS/Face-recognition-Based-Student-Attendance-System/blob/main/LICENSE) file for details.

Project Link: [Face Recognition Based Student Attendance System](https://github.com/JCSTARS/Face-recognition-Based-Student-Attendance-System)

## Reference
facenet-pytorch (https://github.com/timesler/facenet-pytorch)

