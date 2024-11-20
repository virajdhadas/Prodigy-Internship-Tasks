import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# Path to manually split dataset
train_path = 'D:/ProdigyInfotech Internship Task/dataset/train'
val_path = 'D:/ProdigyInfotech Internship Task/dataset/val'

# Step 1: Data Preparation
data_gen = ImageDataGenerator(rescale=1./255)  # No validation split; already manually split

train_data = data_gen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = data_gen.flow_from_directory(
    val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Step 2: Build the Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')  # Dynamically adjust for the number of classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Train the Model
model.fit(train_data, validation_data=val_data, epochs=5)

# Step 4: Calorie Mapping
calorie_map = {
    "pizza": 266,
    "apple_pie": 237
}

def get_calorie_estimation(food_class):
    return calorie_map.get(food_class, "Unknown")

# Step 5: Prediction Function
def predict_food_and_calories(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Load image
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = list(train_data.class_indices.keys())[predicted_class_index]  # Map index to class name

    calories = get_calorie_estimation(predicted_class)
    return predicted_class, calories

# Example Usage
test_image_path = 'D:/ProdigyInfotech Internship Task/dataset/test_image.jpg'
food, calories = predict_food_and_calories(test_image_path)
print(f"Food: {food}, Estimated Calories: {calories}")
