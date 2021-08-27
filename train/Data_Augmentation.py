import secrets
import torchvision.transforms as transforms


contrast = [i/10 for i in range(3, 7)]
factor = [i/100 for i in range(11, 23)]
brightness_factor = secrets.choice(factor)
contrast_factor = secrets.choice(contrast)
saturation_factor = secrets.choice(factor)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(degrees= (-7, 7)),
    transforms.Grayscale(3),
    transforms.ColorJitter(brightness=brightness_factor, contrast=contrast_factor, saturation=saturation_factor, hue=.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.Grayscale(3),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
