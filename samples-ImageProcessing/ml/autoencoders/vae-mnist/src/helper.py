from datetime import datetime
import tensorflow as tf
import numpy as np
  
#########################################################
def lm(msg, end='\n'):
    tm = datetime.now().strftime("%H:%M:%S")
    print(f'{tm} {msg}', end=end)
#########################################################
def save_images_as_grid(images, output_path, items_per_row):
    num_images = len(images)
    num_rows = int(np.ceil(num_images / items_per_row))
    image_height, image_width, _ = images[0].shape
    big_image_height = num_rows * image_height
    big_image_width = items_per_row * image_width
    big_image = np.zeros((big_image_height, big_image_width, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        if isinstance(img, tf.Tensor):
            img = img.numpy()
        row = i // items_per_row
        col = i % items_per_row
        y_start = row * image_height
        y_end = y_start + image_height
        x_start = col * image_width
        x_end = x_start + image_width
        big_image[y_start:y_end, x_start:x_end, :] = (img * 255).astype(np.uint8)

    big_image = tf.convert_to_tensor(big_image)

    tf.keras.preprocessing.image.save_img(output_path, big_image, file_format='jpeg', quality=100)
#########################################################

