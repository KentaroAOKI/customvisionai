import tensorflow as tf
import numpy as np
from PIL import Image

class Classification():
    network_input_size = 227
    output_node = 'loss:0'
    input_node = 'Placeholder:0'

    graph_def = tf.GraphDef()
    labels = []

    def __init__(self, model_pb_file, labels_file):
        """Classes for using https://www.customvision.ai/
        
        Arguments:
            model_pb_file {string} -- tensorflow protocol buffer model
            labels_file {string} -- classification labels
        """

        with tf.gfile.FastGFile(model_pb_file, 'rb') as f:
            self.graph_def.ParseFromString(f.read())
        with open(labels_file, 'rt') as lf:
            for l in lf:
                self.labels.append(l.strip())

    def resize_to_n_square(self, image, size):
        w,h = image.size
        min_dim = min(w,h) // 1.2
        startx = w // 2 - (min_dim // 2)
        starty = h // 2 - (min_dim // 2)
        crop_image = image.crop((startx, starty, startx + min_dim, starty + min_dim))
        return crop_image.resize((size, size), Image.BILINEAR)

    def predict(self, image):
        """Predict image classification
        
        Arguments:
            image {PIL Image} -- Image to be classified
        
        Returns:
            list -- Score each label
        """

        tf.reset_default_graph()
        tf.import_graph_def(self.graph_def, name='')

        with tf.Session() as sess:
            prob_tensor = sess.graph.get_tensor_by_name(self.output_node)
            bgr_image = None
            if(image.mode != "RGB"):
                image.convert("RGB")
            # resize
            resize_image = self.resize_to_n_square(image, self.network_input_size)
            # RGB -> BGR
            r,g,b = np.array(resize_image).T
            bgr_image = np.array([b,g,r]).transpose()
            # predict
            predictions, = sess.run(prob_tensor, {self.input_node: [bgr_image] })

            result = []
            idx = 0
            for p in predictions:
                result.append([p, self.labels[idx]])
                idx += 1       
            return result

if __name__ == "__main__":
    model_filename = 'model.pb'
    labels_filename = 'labels.txt'
    image_filename = 'test.png' 

    # create custom vision object for classification
    customvision = Classification(model_filename, labels_filename)
    # read image from file
    image = Image.open(image_filename)
    # predict image
    result = customvision.predict(image)
    # print score
    print(result)
