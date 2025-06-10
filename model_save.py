import torch
from train import model

#Run this to save model

checkpoint_path = "resnet_captcha_model.pth"
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved to {checkpoint_path}")


from IPython.display import FileLink
FileLink("resnet_captcha_model.pth")
