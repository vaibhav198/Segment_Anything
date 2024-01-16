import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from segment_anything import sam_model_registry, SamPredictor

class SegmentationModel():
    def __init__(self, model_path = './models/sam_vit_h_4b8939.pth', MODEL_TYPE = "vit_h"):
        self.sam = sam_model_registry[MODEL_TYPE](checkpoint=model_path).to(device=DEVICE)

    def pred(self, image, input_box):
        predictor = SamPredictor(self.sam)
        predictor.set_image(image)
        mask, _, _ = predictor.predict(
                        point_coords = None,
                        point_labels = None,
                        box=input_box[None, :],
                        multimask_output=False)
        return mask