import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from torch.autograd import Variable
from torchvision import transforms

from CustomCNN import BasicNet, ImprovedNet, ImprovedNetLite

class_names = [
    'Audi', 'Hyundai_Creta', 'Mahindra_Scorpio',
    'Rolls_Royce', 'Swift', 'Tata_Safari', 'Toyota_Innova'
]


def predict_image(classifier, image_path):
    # Set the classifier model to evaluation mode
    classifier.eval()

    # Apply the same transformations as we did for the training images
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])

    # Preprocess the image
    pil_image = np.array(Image.open(image_path))
    image_tensor = transformation(pil_image).float()

    # Add an extra batch dimension since pytorch treats all inputs as batches
    image_tensor = image_tensor.unsqueeze_(0)

    # Turn the input into a Variable
    input_features = Variable(image_tensor)

    # Predict the class of the image
    output = classifier(input_features)
    index = output.data.numpy().argmax()
    return index


# Initialize Flask app and load model
app = Flask(__name__)
CORS(app)


# Define API endpoint for image classification
@app.route('/classify_image', methods=['POST'])
@cross_origin()
def classify_image():
    image_path = request.get_json()["image_path"]

    try:
        predicted_index = predict_image(model, image_path)
        predicted_class = class_names[predicted_index]
        return jsonify({'class': predicted_class})
    except Exception as e:
        print(f'Error during prediction: {e}')
        return jsonify({'error': 'An error occurred during prediction'}), 500


if __name__ == '__main__':
    port = 9888
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(class_names)
    model = ImprovedNetLite(num_classes)

    model_list = ['BasicNet', 'ImprovedNet', 'ImprovedNetLite']
    current_model = model_list[2]
    batch_size_value = 320
    model_file = 'model_store_' + str(batch_size_value) + '/cnn_car_' + current_model + '.pt'

    model.load_state_dict(torch.load(model_file))
    app.run(port=port, debug=True)
