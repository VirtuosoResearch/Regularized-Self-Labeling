import argparse
import cxr_dataset as CXR
import eval_model as E
import model as M

parser = argparse.ArgumentParser()
parser.add_argument("--reg_method", type = str, default = "constraint")
parser.add_argument("--reg_norm", type = str, default = "frob")
parser.add_argument("--reg_extractor", type = float, default = 1.0)
parser.add_argument("--reg_predictor", type = float, default = 1.0)
parser.add_argument("--scale_factor", type=float, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--device", type=int, default=0)

parser.add_argument("--label_smooth", action="store_true")
parser.add_argument("--smooth_alpha", type=float, default=0.2)
args = parser.parse_args()

# you will need to customize PATH_TO_IMAGES to where you have uncompressed
# NIH images
PATH_TO_IMAGES = "./images/"
WEIGHT_DECAY = args.weight_decay
LEARNING_RATE = 0.01
preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, args,
        lambda_extractor = args.reg_extractor, 
        lambda_pred_head = args.reg_predictor, 
        scale_factor = args.scale_factor, name = "layerwise")
