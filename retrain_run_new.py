import numpy as np
import tensorflow as tf
import urllib
import sys

modelFullPath = '/var/www/html/kakao/output_graph.pb'
labelsFullPath = '/var/www/html/kakao/output_labels.txt'


def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(urlroot, file_name):
    answer = None

    urllib.urlretrieve(urlroot, "/var/www/html/kakao/img.jpg")
    imagePath = '/var/www/html/kakao/img.jpg'

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
	
	file_name = '/var/www/html/kakao/'+file_name
	out_f = open(file_name, 'w')
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            data = '%s (score = %.5f )\n' % (human_string, score)
            print('%s (score = %.5f )' % (human_string, score))
            out_f.write(data)

        answer = labels[top_k[0]]
        return answer


if __name__ == '__main__':
    urlroot = sys.argv[1]
    file_name  = sys.argv[2]
    run_inference_on_image(urlroot, file_name)
