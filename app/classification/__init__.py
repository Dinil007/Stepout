"""Person Classification Pipeline for Team/Role Identification.

Modules:
    config: Configuration dataclasses and constants
    dataset: Dataset preparation, quality filtering, splitting, augmentation
    data: PyTorch Dataset and DataLoader
    models: Model factory (EfficientNet, MobileNet, ConvNeXt, ResNet)
    trainer: Training engine with metrics and TensorBoard
    inference: Inference engine for real-time classification
"""