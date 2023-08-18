# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import insightface
import onnxruntime
from insightface.app import FaceAnalysis
import cv2
import gfpgan
import tempfile
import time

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.face_swapper = insightface.model_zoo.get_model('cache/inswapper_128.onnx', providers=onnxruntime.get_available_providers())
        self.face_enhancer = gfpgan.GFPGANer(model_path='cache/GFPGANv1.4.pth', upscale=1)
        self.face_analyser = FaceAnalysis(name='buffalo_l')
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    
    def get_face(self, img_data):
        analysed = self.face_analyser.get(img_data)
        try:
            largest = max(analysed, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return largest
        except:
            print("No face found") 
            return None
        
    def predict(
        self,
        target_image: Path = Input(description="Target/base image"),
        swap_image: Path = Input(description="Swap/source image")
    ) -> Path:
        """Run a single prediction on the model"""
        try:
            frame = cv2.imread(str(target_image))
            face = self.get_face(frame)
            source_face = self.get_face(cv2.imread(str(swap_image)))
            try:
                print(frame.shape, face.shape, source_face.shape)
            except:
                print("printing shapes failed.")
            result = self.face_swapper.get(frame, face, source_face, paste_back=True)
            
            _, _, result = self.face_enhancer.enhance(
                result,
                paste_back=True
            )
            out_path = Path(tempfile.mkdtemp()) / f"{str(int(time.time()))}.jpg"
            cv2.imwrite(str(out_path), result)
            return out_path
        except Exception as e:
            print(f"{e}")
            return None

    