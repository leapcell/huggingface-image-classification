from transformers import AutoImageProcessor, ResNetForImageClassification

MODEL_PATH = "./model"

model_name = "microsoft/resnet-50"

model = ResNetForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Save to the specified path
model.save_pretrained(MODEL_PATH)
processor.save_pretrained(MODEL_PATH)