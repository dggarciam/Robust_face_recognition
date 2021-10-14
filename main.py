import argparse



def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_path", type=str, default="",
                        help="output folder name")
    parser.add_argument("--model_path", default=0,
                        help="model path, if 0 model will be trained using LogisticRegression")
    parser.add_argument("--known_images_path", type=str, default=0,
                        help="Recalculate embeddings over all images in the path")
    parser.add_argument("--thresh", type=float, default=0.7,
                        help="remove detections with confidence below this value")
    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    
    
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

if __name__ == '__main__':

    args = parser()
    check_arguments_errors(args)