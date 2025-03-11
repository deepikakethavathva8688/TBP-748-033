import torch
from torch import nn, optim
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import numpy as np

# Streamlit configuration
st.set_page_config(page_title="Artistic Fusion: Enhanced Style Transfer", layout="centered")

# Custom styling for Streamlit
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.imgur.com/wsMXixS.jpeg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.8);
}
h1, h2, h3, h4, h5, p, label, div, span {
    color: black !important;
}
button[data-testid="stbutton"] {
    color: white !important;
    background-color: black !important;
    border-radius: 8px;
    font-weight: bold;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the VGG19 model for feature extraction
def load_vgg_model():
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    return vgg

# Preprocessing function to resize and normalize images
def preprocess_image(image, size=256):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Function to deprocess image (convert tensor back to PIL image)
def deprocess_image(tensor):
    image = tensor.squeeze(0).cpu().detach()
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    image = image.numpy().transpose(1, 2, 0) * 255
    return Image.fromarray(image.astype(np.uint8))

# Compute Gram Matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Content and Style Loss computation
def compute_loss(target_features, content_features, style_features, style_grams, content_weight=1, style_weight=1e5):
    # Content loss is still calculated using the conv4_2 feature map
    content_loss = content_weight * torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    style_loss = 0
    for layer in style_grams:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        style_loss += torch.mean((target_gram - style_gram) ** 2)

    total_loss = content_loss + style_loss * style_weight
    return total_loss
# Function to extract features from the VGG19 model
def get_features(image, model, layers=None):
    if layers is None:
        # Extracting features from more layers to capture both content and style details
        layers = {
            '0': 'conv1_1',  # Style features from early layers (low-level details)
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content representation (mid-level features)
            '28': 'conv5_1',  # Style features from deeper layers (high-level details)
            '33': 'conv5_2',  # Additional layers for richer style features
            '36': 'conv5_3'   # Even deeper style layers for complex textures
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Style Transfer Function
def style_transfer(content_tensor, style_tensor, model, device, max_steps=2000):
    content_features = get_features(content_tensor, model)
    style_features = get_features(style_tensor, model)

    # Gram matrices for style features
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Target image initialized as content image
    target = content_tensor.clone().requires_grad_(True).to(device)

    optimizer = optim.Adam([target], lr=0.003)

    progress_bar = st.progress(0)

    for step in range(max_steps):
        target_features = get_features(target, model)

        # Calculate total loss
        loss = compute_loss(target_features, content_features, style_features, style_grams)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar and display progress
        if step % 50 == 0:
            progress_bar.progress(int((step / max_steps) * 100))

    progress_bar.progress(100)  # Complete the progress bar
    return target

# Streamlit UI
st.title("🎨 ARTISTIC FUSION")
st.write("Upload a *content image* and a *style image*, and let the magic of neural style transfer create something extraordinary!")

# File uploaders
col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], key="content_file_uploader")
    if content_file:
        content_image = Image.open(content_file).convert("RGB").resize((256, 256))
        st.image(content_image, caption="Content Image", use_container_width=True)

with col2:
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="style_file_uploader")
    if style_file:
        style_image = Image.open(style_file).convert("RGB").resize((256, 256))
        st.image(style_image, caption="Style Image", use_container_width=True)

# Style transfer button
if content_file and style_file:
    if st.button("Apply Style Transfer"):
        st.write("Processing images...")

        # Check if CUDA is available and set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"Using device: {device}")

        # Load the VGG model
        model = load_vgg_model().to(device)

        # Preprocess images
        content_tensor = preprocess_image(content_image).to(device)
        style_tensor = preprocess_image(style_image).to(device)

        # Perform style transfer
        output_tensor = style_transfer(content_tensor, style_tensor, model, device)

        # Convert output tensor to image
        output_image = deprocess_image(output_tensor)
        st.image(output_image, caption="Stylized Image", width=356)
