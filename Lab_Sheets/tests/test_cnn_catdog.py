def test_cnn_model_fit():
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from keras.preprocessing.image import ImageDataGenerator
    import numpy as np

    train_set = ImageDataGenerator().flow_from_directory('path_to_train_set', target_size=(150, 150), batch_size=32, class_mode='binary')
    test_set = ImageDataGenerator().flow_from_directory('path_to_test_set', target_size=(150, 150), batch_size=32, class_mode='binary')

    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=128, activation='relu'))
    cnn_model.add(Dense(units=1, activation='sigmoid'))

    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    cnn_model.fit(train_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)

    loss, accuracy = cnn_model.evaluate(test_set)
    assert accuracy > 0.8, "Model accuracy is below 80%"

def test_import_pil():
    try:
        from PIL import Image
    except ImportError:
        assert False, "Pillow is not installed. Please install it to run the tests."