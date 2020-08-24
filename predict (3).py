from utilities import *
from train import checkpoint

model.class_to_idx = checkpoint['class_to_idx']
model, class_to_idx = load_checkpoint('checkpoint.pth')

probs, classes = predict(args.image_path, model, args.topk)

print ('Classes: ', classes)
print('Probability: ', probs)
