from flask import Flask, request, jsonify
from predict import predict_animal
from PIL import Image
import io

app = Flask(__name__)

# List of animals (same as your list)
animals = [
    "antelope", "badger", "bat", "bear", "bee", "beetle", "bison", "boar", "butterfly", "cat",
    "caterpillar", "chimpanzee", "cockroach", "cow", "coyote", "crab", "crow", "deer", "dog",
    "dolphin", "donkey", "dragonfly", "duck", "eagle", "elephant", "flamingo", "fly", "fox",
    "goat", "goldfish", "goose", "gorilla", "grasshopper", "hamster", "hare", "hedgehog",
    "hippopotamus", "hornbill", "horse", "hummingbird", "hyena", "jellyfish", "kangaroo",
    "koala", "ladybugs", "leopard", "lion", "lizard", "lobster", "mosquito", "moth", "mouse",
    "octopus", "okapi", "orangutan", "otter", "owl", "ox", "oyster", "panda", "parrot",
    "pelecaniformes", "penguin", "pig", "pigeon", "porcupine", "possum", "raccoon", "rat",
    "reindeer", "rhinoceros", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake",
    "sparrow", "squid", "squirrel", "starfish", "swan", "tiger", "turkey", "turtle", "whale",
    "wolf", "wombat", "woodpecker", "zebra"
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image.save('temp.jpg')  # Save temporarily for prediction
    animal_class = predict_animal('temp.jpg')
    animal_name = animals[animal_class]

    return jsonify({'animal': animal_name})

if __name__ == '__main__':
    app.run(debug=True)